"""
SLTrust-DFL Experiment Dashboard (TUI)
"""
import sys
import subprocess
import threading
import time
import os
from pathlib import Path

os.environ.setdefault('PYTHONUTF8', '1')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from datetime import datetime
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

PROJ_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJ_DIR))

try:
    from utils import check_all_datasets, download_dataset
except ModuleNotFoundError as exc:
    if exc.name != 'utils':
        raise
    raise SystemExit(
        "Project is incomplete: missing the local 'utils/' package.\n"
        f"Expected: {PROJ_DIR / 'utils' / '__init__.py'}\n"
        "Upload/sync the complete SLTrust_DFL_Experiments directory; "
        "do not install a PyPI package named 'utils'."
    ) from exc

DATA_DIR = str(PROJ_DIR / 'data')
console = Console(width=80)

EXPERIMENTS = {
    'A': {
        'label': 'MVP 验证实验 — MNIST',
        'items': [
            ('A1', 'mvp',         'MVP 串行（4方法×3攻击，顺序跑）'),
            ('A2', 'mvp_parallel','MVP 并行（推荐，6并发）'),
            ('A3', 'mvp_plot',    'MVP 合并出图'),
        ]
    },
    'B': {
        'label': 'MVP 验证实验 — CIFAR-10',
        'items': [
            ('B1', 'mvp_cifar10',         'CIFAR-10 串行'),
            ('B2', 'mvp_cifar10_parallel','CIFAR-10 并行（推荐）'),
            ('B3', 'mvp_cifar10_plot',    'CIFAR-10 合并出图'),
        ]
    },
    'C': {
        'label': '主表实验 — 8方法×8攻击',
        'items': [
            ('C1', 'main_table',          '主表 MNIST 串行'),
            ('C2', 'main_table_parallel', '主表 MNIST 并行（多轮多seed）'),
            ('C3', 'main_table_plot',     '主表 MNIST 合并出图'),
            ('C4', 'main_table_cifar10',          '主表 CIFAR-10 串行'),
            ('C5', 'main_table_cifar10_parallel', '主表 CIFAR-10 并行（多轮多seed）'),
            ('C6', 'main_table_cifar10_plot',     '主表 CIFAR-10 合并出图'),
        ]
    },
    'D': {
        'label': 'τ 演化实验',
        'items': [
            ('D1', 'tau_mnist',   'τ 演化 MNIST（contra+sltrust × 7攻击）'),
            ('D2', 'tau_cifar10', 'τ 演化 CIFAR-10'),
        ]
    },
    'E': {
        'label': '消融实验 — SLTrust 内部变体',
        'items': [
            ('E1', 'ablation_mnist',   '消融 MNIST（4变体×2攻击，多轮多seed）'),
            ('E2', 'ablation_cifar10', '消融 CIFAR-10'),
        ]
    },
    'F': {
        'label': '参数敏感性实验',
        'items': [
            ('F1', 'sens_burnin',  'Burn-in warmup 扫描（warmup=20/40/60/80）'),
            ('F2', 'sens_osc',     'Oscillating period 扫描（period=5/10/20）'),
            ('F3', 'sens_plot',    '敏感性实验出图'),
            ('F4', 'sens_sl_grid', 'S/L 超参网格扫描（S∈{2,3,5,10} × L∈{5,10,20,40}）'),
        ]
    },
    'G': {
        'label': '一键全部论文实验',
        'items': [
            ('G1', 'paper_all', '选实验类型+数据集+轮数，统一并行调度'),
        ]
    },
}


# ──────────────────────────────────────────────────────────
# GPU 信息
# ──────────────────────────────────────────────────────────

def get_gpu_info():
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(',')
        return {
            'mem_used': int(parts[0].strip()),
            'mem_total': int(parts[1].strip()),
            'util': int(parts[2].strip()),
            'temp': int(parts[3].strip()),
        }
    except Exception:
        return None


def render_gpu_panel():
    info = get_gpu_info()
    if info is None:
        return Panel("[dim]GPU 信息不可用[/dim]", title="GPU", border_style="dim")
    pct = info['mem_used'] / info['mem_total']
    bar_len = 20
    filled = int(bar_len * pct)
    bar = "#" * filled + "." * (bar_len - filled)
    color = "green" if pct < 0.7 else ("yellow" if pct < 0.9 else "red")
    mem_text = Text()
    mem_text.append(f"MEM  [{bar}] ", style="white")
    mem_text.append(f"{info['mem_used']}/{info['mem_total']} MB ({pct:.0%})", style=color)
    util_pct = info['util'] / 100
    u_filled = int(bar_len * util_pct)
    u_bar = "#" * u_filled + "." * (bar_len - u_filled)
    u_color = "green" if util_pct < 0.5 else ("yellow" if util_pct < 0.85 else "red")
    util_text = Text()
    util_text.append(f"UTIL [{u_bar}] ", style="white")
    util_text.append(f"{info['util']}%", style=u_color)
    util_text.append(f"   Temp {info['temp']}C",
                     style="green" if info['temp'] < 75 else "yellow")
    content = Text.assemble(mem_text, "\n", util_text)
    return Panel(content, title="[bold cyan]GPU[/bold cyan]", border_style="cyan")


# ──────────────────────────────────────────────────────────
# 结果扫描
# ──────────────────────────────────────────────────────────

def scan_results():
    results_dir = PROJ_DIR / 'results'
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob('*.png'), key=lambda f: f.stat().st_mtime, reverse=True)


def render_results_table():
    files = scan_results()
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta", expand=True)
    table.add_column("文件名", style="cyan", no_wrap=True)
    table.add_column("大小", justify="right", style="green")
    table.add_column("时间", style="dim")
    if not files:
        table.add_row("[dim]暂无结果文件[/dim]", "", "")
    else:
        results_dir = PROJ_DIR / 'results'
        for f in files[:8]:
            try:
                rel = str(f.relative_to(results_dir))
            except ValueError:
                rel = f.name
            size = f"{f.stat().st_size / 1024:.1f} KB"
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%m-%d %H:%M')
            table.add_row(rel, size, mtime)
    return Panel(table, title=f"[bold magenta]已有结果 ({len(files)} 个)[/bold magenta]",
                 border_style="magenta")


# ──────────────────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────────────────

def render_dataset_panel(status):
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("数据集", style="bold", width=12)
    table.add_column("状态")
    for name, ok in status.items():
        table.add_row(name.upper(), "[green]OK[/green]" if ok else "[red]缺失[/red]")
    all_ok = all(status.values())
    return Panel(table,
                 title="[bold green]数据集状态[/bold green]" if all_ok
                       else "[bold yellow]数据集状态  (有缺失)[/bold yellow]",
                 border_style="green" if all_ok else "yellow")


def check_and_prompt_datasets():
    status = check_all_datasets(DATA_DIR)
    console.print(render_dataset_panel(status))
    missing = [name for name, ok in status.items() if not ok]
    if not missing:
        return
    console.print(f"[yellow]缺失: {', '.join(m.upper() for m in missing)}[/yellow]")
    if Confirm.ask("是否下载缺失数据集?", default=True):
        for name in missing:
            console.print(f"\n[cyan]下载 {name.upper()}...[/cyan]")
            script = (f"import sys; sys.path.insert(0, r'{PROJ_DIR}'); "
                      f"from utils import download_dataset; "
                      f"download_dataset('{name}', r'{DATA_DIR}')")
            subprocess.run([sys.executable, '-u', '-c', script], cwd=str(PROJ_DIR))
        console.print(render_dataset_panel(check_all_datasets(DATA_DIR)))


# ──────────────────────────────────────────────────────────
# 实验运行器
# ──────────────────────────────────────────────────────────

class ExperimentRunner:

    def __init__(self, exp_key, label, extra_env=None):
        self.exp_key = exp_key
        self.label = label
        self.extra_env = extra_env or {}
        self.running = False
        self.done = False
        self.returncode = None
        self.output_lines = []
        self.start_time = None
        self.end_time = None
        self.proc = None
        self.log_path = None
        self._lock = threading.Lock()
        self._log_file = None

    def start(self):
        self.running = True
        self.start_time = datetime.now()
        # 落盘日志：按实验类型和日期分子目录
        # exp_key 形如 main_table_single / ablation_cifar10_single / tau_mnist_single
        # 提取前缀作为子目录：main_table / ablation / tau / tau_decay / sensitivity
        key = self.exp_key
        if 'main_table' in key:
            sub = 'main_table_cifar10' if 'cifar10' in key else 'main_table'
        elif 'ablation' in key:
            sub = 'ablation_cifar10' if 'cifar10' in key else 'ablation'
        elif 'tau_decay' in key:
            sub = 'tau_decay_cifar10' if 'cifar10' in key else 'tau_decay'
        elif 'tau' in key:
            sub = 'tau_cifar10' if 'cifar10' in key else 'tau'
        elif 'sensitivity' in key:
            sub = 'sensitivity'
        else:
            sub = key
        log_dir = PROJ_DIR / 'logs' / sub / f"{self.start_time:%Y%m%d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = self.label.replace('|', '-').replace(' ', '_')[:60]
        self.log_path = log_dir / f"{safe_label}_{self.start_time:%H%M%S}.log"
        self._log_file = open(self.log_path, 'w', encoding='utf-8', buffering=1)

        cmd = [sys.executable, '-u', str(PROJ_DIR / 'run_experiments.py'),
               '--experiment', self.exp_key]
        env = os.environ.copy()
        env.update(self.extra_env)
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=str(PROJ_DIR), bufsize=0)
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        import io as _io
        raw = (self.proc.stdout.buffer
               if isinstance(self.proc.stdout, _io.TextIOBase)
               else self.proc.stdout)
        buf = ""
        while True:
            try:
                chunk = raw.read1(4096) if hasattr(raw, 'read1') else raw.read(1)
            except Exception:
                break
            if not chunk:
                break
            buf += chunk.decode('utf-8', errors='replace')
            while '\n' in buf or '\r' in buf:
                nl = buf.find('\n')
                cr = buf.find('\r')
                if nl == -1:
                    idx, is_cr = cr, True
                elif cr == -1:
                    idx, is_cr = nl, False
                else:
                    idx, is_cr = (cr, True) if cr < nl else (nl, False)
                line = buf[:idx].rstrip('\r\n')
                buf = buf[idx + 1:]
                if not line:
                    continue
                # 落盘
                if self._log_file:
                    try:
                        self._log_file.write(line + '\n')
                    except Exception:
                        pass
                with self._lock:
                    if is_cr and self.output_lines:
                        self.output_lines[-1] = line
                    else:
                        self.output_lines.append(line)
                        if len(self.output_lines) > 200:
                            self.output_lines = self.output_lines[-200:]
        self.proc.wait()
        self.returncode = self.proc.returncode
        self.running = False
        self.done = True
        self.end_time = datetime.now()
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass

    def stop(self):
        if self.proc and self.running:
            self.proc.terminate()

    def get_last_lines(self, n=8):
        with self._lock:
            return list(self.output_lines[-n:])

    def elapsed(self):
        if not self.start_time:
            return ""
        s = int((datetime.now() - self.start_time).total_seconds())
        return f"{s // 60:02d}:{s % 60:02d}"

    def status_str(self):
        if self.done:
            return "[green]完成[/green]" if self.returncode == 0 else f"[red]失败({self.returncode})[/red]"
        if self.running:
            return f"[cyan]运行中 {self.elapsed()}[/cyan]"
        return "[dim]等待[/dim]"


# ──────────────────────────────────────────────────────────
# 单实验运行（带 Live 输出）
# ──────────────────────────────────────────────────────────

def run_single(exp_key, label, extra_env=None):
    job = ExperimentRunner(exp_key, label, extra_env=extra_env)
    job.start()
    try:
        with Live(console=console, refresh_per_second=4, screen=False) as live:
            while job.running:
                lines = job.get_last_lines(14)
                log_text = Text()
                for line in lines:
                    if 'ERROR' in line or 'Error' in line:
                        log_text.append(line + "\n", style="red")
                    elif 'acc=' in line.lower() or '%' in line:
                        log_text.append(line + "\n", style="green")
                    elif 'Round' in line or 'round' in line or '---' in line:
                        log_text.append(line + "\n", style="cyan")
                    elif line.startswith('=') or line.startswith('['):
                        log_text.append(line + "\n", style="yellow")
                    else:
                        log_text.append(line + "\n", style="dim white")
                layout = Layout()
                layout.split_column(
                    Layout(render_gpu_panel(), size=4),
                    Layout(Panel(log_text,
                                 title=f"[bold cyan]{label}  {job.elapsed()}[/bold cyan]",
                                 border_style="cyan")),
                    Layout(Panel("[dim]Ctrl+C 中断[/dim]", border_style="dim"), size=3),
                )
                live.update(layout)
                time.sleep(0.25)
    except KeyboardInterrupt:
        job.stop()
        console.print("\n[yellow]已中断[/yellow]")
        return

    if job.returncode == 0:
        console.print(Rule(f"[green]✓ {label} 完成  耗时 {job.elapsed()}[/green]"))
    else:
        console.print(Rule(f"[red]✗ {label} 失败 (exit {job.returncode})[/red]"))
    for line in job.get_last_lines(6):
        console.print(f"  [dim]{line}[/dim]")


# ──────────────────────────────────────────────────────────
# 并行运行（复刻 SAMA 的 run_parallel 逻辑）
# ──────────────────────────────────────────────────────────

def run_parallel(jobs: list, max_concurrent: int = 6):
    """最多同时跑 max_concurrent 个唯一子进程，其余排队。"""
    unique_jobs = []
    seen = set()
    for job in jobs:
        identity = (job.exp_key, tuple(sorted(job.extra_env.items())))
        if identity in seen:
            continue
        seen.add(identity)
        unique_jobs.append(job)
    if len(unique_jobs) != len(jobs):
        console.print(f"[yellow]调度去重：跳过 {len(jobs) - len(unique_jobs)} 个重复 job[/yellow]")
    jobs = unique_jobs
    pending = list(jobs)   # 还未启动
    started = []           # 已启动（running 或 done）

    def _fill_slots():
        while pending and sum(1 for j in started if j.running) < max_concurrent:
            j = pending.pop(0)
            j.start()
            started.append(j)

    _fill_slots()

    interrupted = False
    total_jobs = len(jobs)
    wall_start = datetime.now()
    eta_str = ""
    last_done_count = 0
    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                _fill_slots()

                done_jobs    = [j for j in started if j.done]
                running_jobs = [j for j in started if j.running]
                waiting_jobs = [j for j in started if not j.done and not j.running] + pending

                # 只在有新 job 完成时重新计算 ETA
                if len(done_jobs) != last_done_count:
                    last_done_count = len(done_jobs)
                    durations = [
                        (j.end_time - j.start_time).total_seconds()
                        for j in done_jobs if j.start_time and j.end_time
                    ]
                    if durations:
                        avg_sec = sum(durations) / len(durations)
                        remaining = len(pending) + len(running_jobs)
                        eta_sec = int(avg_sec * remaining / max(max_concurrent, 1))
                        h, m = divmod(eta_sec // 60, 60)
                        eta_str = f"  ETA {h}h{m:02d}m" if h else f"  ETA {m}m"

                renderables = []
                for job in running_jobs:
                    lines = job.get_last_lines(8)
                    log_text = Text()
                    for line in lines:
                        if 'ERROR' in line or 'Error' in line:
                            log_text.append(f"  {line}\n", style="red")
                        elif 'acc=' in line.lower() or '%' in line:
                            log_text.append(f"  {line}\n", style="green")
                        elif 'Round' in line or '---' in line:
                            log_text.append(f"  {line}\n", style="cyan")
                        else:
                            log_text.append(f"  {line}\n", style="dim white")
                    renderables.append(Panel(
                        log_text,
                        title=f"[bold cyan]{job.label}  {job.elapsed()}[/bold cyan]",
                        border_style="cyan", padding=(0, 1),
                    ))

                if done_jobs or waiting_jobs:
                    table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
                    table.add_column("实验", style="white")
                    table.add_column("状态", width=18)
                    for job in done_jobs:
                        table.add_row(job.label, job.status_str())
                    for job in pending:
                        table.add_row(job.label, "[dim]排队[/dim]")
                    renderables.append(Panel(table, border_style="dim", padding=(0, 1)))

                slots_text = f"运行: {len(running_jobs)}  完成: {len(done_jobs)}/{total_jobs}  排队: {len(pending)}{eta_str}"
                from rich.console import Group as RichGroup
                jobs_panel = Panel(
                    RichGroup(*renderables) if renderables else Text(""),
                    title=f"[bold cyan]实验进度  {slots_text}[/bold cyan]",
                    border_style="cyan",
                )
                layout = Layout()
                layout.split_column(Layout(render_gpu_panel(), size=4), Layout(jobs_panel))
                live.update(layout)
                time.sleep(0.5)

                if all(j.done for j in started) and not pending:
                    break

    except KeyboardInterrupt:
        interrupted = True
        for j in started:
            j.stop()
        console.print("\n[yellow]已中断[/yellow]")

    if not interrupted:
        failed = [job for job in started if job.returncode != 0]
        style = "bold red" if failed else "bold green"
        title = f"{len(failed)} 个任务失败" if failed else "全部完成"
        console.print(Rule(f"[{style}]{title}[/{style}]"))
        for job in started:
            mark = "[green]✓[/green]" if job.returncode == 0 else "[red]✗[/red]"
            console.print(f"  {mark} {job.label}  耗时 {job.elapsed()}")
        return not failed
    return False




ALL_ATTACKS = [
    ('none',         'No Attack'),
    ('sign_flip',    'Sign Flip'),
    ('gaussian',     'Gaussian Noise'),
    ('burn_in',      'Burn-in'),
    ('oscillating',  'Oscillating'),
    ('slow_poison',  'Slow Poison'),
    ('gradual_flip', 'Gradual Flip'),
    ('backdoor',     'Backdoor'),
]

ALL_METHODS = [
    ('fedavg',       'FedAvg'),
    ('sama',         'SAMA'),
    ('balance',      'BALANCE'),
    ('krum',         'Krum'),
    ('trimmed_mean', 'Trimmed Mean'),
    ('contra',       'CONTRA'),
    ('contra_hard',  'CONTRA-Hard'),
    ('sltrust',      'SLTrust'),
]


def choose_attacks_multi():
    """让用户多选要跑的攻击。返回逗号分隔字符串（空字符串=全部）。"""
    console.print()
    console.print("[bold]选择本次要跑的攻击[/bold]（多选，逗号或空格分隔；留空 / a / all = 全部）")
    for idx, (key, label) in enumerate(ALL_ATTACKS, start=1):
        console.print(f"  [cyan]{idx}[/cyan] {label}  [dim]({key})[/dim]")
    raw = Prompt.ask("攻击编号", default="").strip().lower()
    if raw in ('', 'a', 'all'):
        return ''
    tokens = [t for t in raw.replace(',', ' ').split() if t]
    selected = []
    for t in tokens:
        if t.isdigit():
            i = int(t)
            if 1 <= i <= len(ALL_ATTACKS):
                selected.append(ALL_ATTACKS[i - 1][0])
        elif any(t == k for k, _ in ALL_ATTACKS):
            selected.append(t)
    if not selected:
        console.print("[yellow]未识别有效攻击，回退为全部[/yellow]")
        return ''
    seen = set()
    unique = [a for a in selected if not (a in seen or seen.add(a))]
    console.print(f"[green]已选攻击: {', '.join(unique)}[/green]")
    return ','.join(unique)


def choose_methods_multi():
    """让用户多选要跑的方法。返回逗号分隔字符串（空字符串=全部）。"""
    console.print()
    console.print("[bold]选择本次要跑的算法[/bold]（多选，逗号或空格分隔；留空 / a / all = 全部）")
    for idx, (key, label) in enumerate(ALL_METHODS, start=1):
        console.print(f"  [cyan]{idx}[/cyan] {label}  [dim]({key})[/dim]")
    raw = Prompt.ask("算法编号", default="").strip().lower()
    if raw in ('', 'a', 'all'):
        return ''
    tokens = [t for t in raw.replace(',', ' ').split() if t]
    selected = []
    for t in tokens:
        if t.isdigit():
            i = int(t)
            if 1 <= i <= len(ALL_METHODS):
                selected.append(ALL_METHODS[i - 1][0])
        elif any(t == k for k, _ in ALL_METHODS):
            selected.append(t)
    if not selected:
        console.print("[yellow]未识别有效算法，回退为全部[/yellow]")
        return ''
    seen = set()
    unique = [a for a in selected if not (a in seen or seen.add(a))]
    console.print(f"[green]已选算法: {', '.join(unique)}[/green]")
    return ','.join(unique)


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def choose_num_seeds():
    """选择运行几轮（每轮用不同 seed）。返回 seed 列表。"""
    console.print()
    raw = Prompt.ask("[bold]运行轮数[/bold]（1=单次验证，3-5=论文多seed；留空=1）", default="1").strip()
    try:
        n = max(1, min(10, int(raw)))
    except ValueError:
        n = 1
    seeds = DEFAULT_SEEDS[:n]
    if n > len(DEFAULT_SEEDS):
        import random
        rng = random.Random(42)
        seeds = DEFAULT_SEEDS + [rng.randint(1, 9999) for _ in range(n - len(DEFAULT_SEEDS))]
    console.print(f"[green]将跑 {n} 轮，seeds: {seeds}[/green]")
    return seeds


def render_menu():
    table = Table(box=box.ROUNDED, expand=True, show_header=False, padding=(0, 1))
    table.add_column("组", style="bold yellow", width=4)
    table.add_column("编号", style="bold cyan", width=4)
    table.add_column("描述", style="white")
    for gk, group in EXPERIMENTS.items():
        table.add_row(f"[{gk}]", "", f"[bold]{group['label']}[/bold]")
        for idx_key, exp_key, label in group['items']:
            table.add_row("", idx_key, label)
    table.add_row("", "", "")
    table.add_row("[Q]", "quit", "[dim]退出[/dim]")
    return Panel(table, title="[bold white]SLTrust-DFL Experiment Dashboard[/bold white]",
                 border_style="blue")


def main():
    if not sys.stdin.isatty():
        console.print("[red]tui.py requires an interactive terminal.[/red]")
        console.print("[dim]Open PowerShell or CMD and run: python tui.py[/dim]")
        return
    console.clear()
    console.print(render_gpu_panel())
    check_and_prompt_datasets()
    console.print(render_results_table())

    while True:
        console.print()
        console.print(render_menu())
        console.print()
        choice = Prompt.ask("[bold white]输入编号[/bold white]", default="").strip().lower()

        if choice in ('q', 'quit', 'exit'):
            console.print("\n[dim]Bye.[/dim]")
            break

        # ── MVP 并行 ──────────────────────────────────────────────────────
        if choice.upper() in ('A2', 'MVP_PARALLEL', 'B2', 'MVP_CIFAR10_PARALLEL'):
            is_cifar = choice.upper() in ('B2', 'MVP_CIFAR10_PARALLEL')
            exp_single = 'mvp_cifar10_single' if is_cifar else 'mvp_single'
            exp_plot   = 'mvp_cifar10_plot'   if is_cifar else 'mvp_plot'
            plot_label = 'CIFAR-10 合并出图'  if is_cifar else 'MVP 合并出图'
            from experiments.performance.mvp_experiment import MVP_METHODS, MVP_ATTACKS
            jobs = []
            for attack in MVP_ATTACKS:
                for method in MVP_METHODS:
                    jobs.append(ExperimentRunner(
                        exp_single, f"{method} | {attack}",
                        extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack}
                    ))
            if not run_parallel(jobs):
                console.print("[red]存在失败任务，已跳过合并出图。[/red]")
                continue
            console.print(f"\n[cyan]合并出图...[/cyan]")
            run_single(exp_plot, plot_label)
            console.print()
            console.print(render_results_table())
            continue

        # ── 主表并行 ──────────────────────────────────────────────────────
        if choice.upper() in ('C2', 'MAIN_TABLE_PARALLEL', 'C5', 'MAIN_TABLE_CIFAR10_PARALLEL'):
            is_cifar = choice.upper() in ('C5', 'MAIN_TABLE_CIFAR10_PARALLEL')
            exp_single = 'main_table_cifar10_single' if is_cifar else 'main_table_single'
            exp_plot   = 'main_table_cifar10_plot'   if is_cifar else 'main_table_plot'
            plot_label = '主表 CIFAR-10 合并出图'    if is_cifar else '主表 MNIST 合并出图'
            from experiments.performance.main_table_experiment import MAIN_METHODS, MAIN_ATTACKS
            attacks_csv = choose_attacks_multi()
            methods_csv = choose_methods_multi()
            seeds = choose_num_seeds()
            attacks = attacks_csv.split(',') if attacks_csv else [k for k, _ in ALL_ATTACKS]
            methods = methods_csv.split(',') if methods_csv else [k for k, _ in ALL_METHODS]
            jobs = []
            for seed in seeds:
                for attack in attacks:
                    for method in methods:
                        jobs.append(ExperimentRunner(
                            exp_single, f"s{seed}|{method}|{attack}",
                            extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                                       'MVP_SEED': str(seed)}
                        ))
            if not run_parallel(jobs):
                console.print("[red]存在失败任务，已跳过主表出图。[/red]")
                continue
            console.print(f"\n[cyan]按 seed 出图...[/cyan]")
            for seed in seeds:
                run_single(exp_plot, f"{plot_label} seed={seed}", {'MVP_SEED': str(seed)})
            console.print()
            console.print(render_results_table())
            continue

        # ── 消融并行（4变体×2攻击×N轮）────────────────────────────
        if choice.upper() in ('E1', 'ABLATION_MNIST', 'E2', 'ABLATION_CIFAR10'):
            is_cifar = choice.upper() in ('E2', 'ABLATION_CIFAR10')
            exp_single = 'ablation_cifar10_single' if is_cifar else 'ablation_single'
            exp_plot   = 'ablation_cifar10_plot'   if is_cifar else 'ablation_plot'
            plot_label = '消融 CIFAR-10 出图'       if is_cifar else '消融 MNIST 出图'
            from experiments.performance.ablation_experiment import ABLATION_METHODS, ABLATION_ATTACKS
            seeds = choose_num_seeds()
            jobs = []
            for seed in seeds:
                for attack in ABLATION_ATTACKS:
                    for method in ABLATION_METHODS:
                        jobs.append(ExperimentRunner(
                            exp_single, f"s{seed}|{method}|{attack}",
                            extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                                       'MVP_SEED': str(seed)}
                        ))
            if not run_parallel(jobs):
                console.print("[red]存在失败任务，已跳过消融出图。[/red]")
                continue
            console.print(f"\n[cyan]按 seed 出图...[/cyan]")
            for seed in seeds:
                run_single(exp_plot, f"{plot_label} seed={seed}", {'MVP_SEED': str(seed)})
            console.print()
            console.print(render_results_table())
            continue

        # ── 敏感性实验 F1: Burn-in warmup 扫描 ────────────────────
        if choice.upper() in ('F1', 'SENS_BURNIN'):
            from experiments.performance.sensitivity_experiment import BURNIN_WARMUPS, SENS_METHODS
            seeds = choose_num_seeds()
            jobs = []
            for seed in seeds:
                for warmup in BURNIN_WARMUPS:
                    for method in SENS_METHODS:
                        jobs.append(ExperimentRunner(
                            'sensitivity_single',
                            f"s{seed}|{method}|burnin_w{warmup}",
                            extra_env={'MVP_METHOD': method, 'MVP_ATTACK': 'burn_in',
                                       'MVP_WARMUP': str(warmup), 'MVP_SEED': str(seed)}
                        ))
            if not run_parallel(jobs):
                console.print("[red]存在失败任务，已跳过敏感性出图。[/red]")
                continue
            console.print(f"\n[cyan]出图...[/cyan]")
            run_single('sensitivity_plot', '敏感性出图')
            console.print()
            console.print(render_results_table())
            continue

        # ── 敏感性实验 F2: Oscillating period 扫描 ─────────────────
        if choice.upper() in ('F2', 'SENS_OSC'):
            from experiments.performance.sensitivity_experiment import OSC_PERIODS, SENS_METHODS
            seeds = choose_num_seeds()
            jobs = []
            for seed in seeds:
                for period in OSC_PERIODS:
                    for method in SENS_METHODS:
                        jobs.append(ExperimentRunner(
                            'sensitivity_single',
                            f"s{seed}|{method}|osc_p{period}",
                            extra_env={'MVP_METHOD': method, 'MVP_ATTACK': 'oscillating',
                                       'MVP_PERIOD': str(period), 'MVP_SEED': str(seed)}
                        ))
            if not run_parallel(jobs):
                console.print("[red]存在失败任务，已跳过敏感性出图。[/red]")
                continue
            console.print(f"\n[cyan]出图...[/cyan]")
            run_single('sensitivity_plot', '敏感性出图')
            console.print()
            console.print(render_results_table())
            continue

        # ── 敏感性实验 F3: 仅出图 ─────────────────────────────────
        if choice.upper() in ('F3', 'SENS_PLOT'):
            run_single('sensitivity_plot', '敏感性出图')
            console.print()
            console.print(render_results_table())
            continue

        # ── 敏感性实验 F4: S/L 超参网格扫描 ───────────────────────
        if choice.upper() in ('F4', 'SENS_SL_GRID'):
            from experiments.performance.sensitivity_experiment import SL_S_VALUES, SL_L_VALUES
            seeds = choose_num_seeds()
            jobs = []
            for seed in seeds:
                for S in SL_S_VALUES:
                    for L in SL_L_VALUES:
                        if S >= L:
                            continue
                        jobs.append(ExperimentRunner(
                            'sensitivity_sl_single',
                            f"s{seed}|S={S}|L={L}",
                            extra_env={'MVP_S': str(S), 'MVP_L': str(L), 'MVP_SEED': str(seed)}
                        ))
            if not run_parallel(jobs):
                console.print("[red]存在失败任务，已跳过 S/L 出图。[/red]")
                continue
            console.print(f"\n[cyan]出图...[/cyan]")
            run_single('sensitivity_plot', '敏感性出图（含S/L热力图）')
            console.print()
            console.print(render_results_table())
            continue

        # ── 一键全部论文实验 G1 ───────────────────────────────
        if choice.upper() in ('G1', 'PAPER_ALL'):
            from experiments.performance.ablation_experiment import ABLATION_METHODS, ABLATION_ATTACKS
            MAIN_METHODS = [k for k, _ in ALL_METHODS]
            MAIN_ATTACKS = [k for k, _ in ALL_ATTACKS]

            # 1. 选实验类型
            console.print()
            console.print("[bold]选择要跑的实验类型[/bold]（多选，逗号或空格分隔；留空=全部）")
            EXP_TYPES = [
                ('main',  '主表  （方法×攻击全表）'),
                ('ablation', '消融  （SLTrust内部变体）'),
                ('tau',   'τ演化  （CONTRA vs SLTrust曲线）'),
                ('tau_decay', 'τ衰减  （衰减速率定量柱状图）'),
            ]
            for idx, (key, label) in enumerate(EXP_TYPES, start=1):
                console.print(f"  [cyan]{idx}[/cyan] {label}")
            raw = Prompt.ask("实验类型编号", default="").strip().lower()
            if raw in ('', 'a', 'all'):
                selected_types = [k for k, _ in EXP_TYPES]
            else:
                tokens = [t for t in raw.replace(',', ' ').split() if t]
                selected_types = []
                for t in tokens:
                    if t.isdigit():
                        i = int(t)
                        if 1 <= i <= len(EXP_TYPES):
                            selected_types.append(EXP_TYPES[i - 1][0])
                if not selected_types:
                    selected_types = [k for k, _ in EXP_TYPES]
            console.print(f"[green]已选实验: {', '.join(selected_types)}[/green]")

            # 2. 选数据集
            console.print()
            console.print("[bold]选择数据集[/bold]")
            console.print("  [cyan]1[/cyan] MNIST only")
            console.print("  [cyan]2[/cyan] CIFAR-10 only")
            console.print("  [cyan]3[/cyan] 两者都跑（推荐）")
            ds_raw = Prompt.ask("数据集", default="3").strip()
            if ds_raw == '1':
                datasets = ['mnist']
            elif ds_raw == '2':
                datasets = ['cifar10']
            else:
                datasets = ['mnist', 'cifar10']
            console.print(f"[green]已选数据集: {datasets}[/green]")

            # 3. 选轮数（按数据集分别配置）
            has_single_only = all(t in ('tau', 'tau_decay') for t in selected_types)
            if has_single_only:
                seeds_mnist   = [DEFAULT_SEEDS[0]]
                seeds_cifar10 = [DEFAULT_SEEDS[0]]
                console.print(f"[dim]τ实验为机制可视化，固定跑 1 次，seed={DEFAULT_SEEDS[0]}[/dim]")
            else:
                if 'mnist' in datasets:
                    console.print("[bold]MNIST 轮数[/bold]", end=" ")
                    seeds_mnist = choose_num_seeds()
                else:
                    seeds_mnist = []
                if 'cifar10' in datasets:
                    console.print("[bold]CIFAR-10 轮数[/bold]", end=" ")
                    seeds_cifar10 = choose_num_seeds()
                else:
                    seeds_cifar10 = []

            # 4. 主表：选 methods/attacks
            if 'main' in selected_types:
                attacks_csv = choose_attacks_multi()
                methods_csv = choose_methods_multi()
                sel_attacks = attacks_csv.split(',') if attacks_csv else MAIN_ATTACKS
                sel_methods = methods_csv.split(',') if methods_csv else MAIN_METHODS
            else:
                sel_attacks, sel_methods = [], []

            # 5. 汇总所有 job
            # tau/tau_decay/ablation 是机制可视化或固定对比，始终只跑 1 次（固定 seed=42）
            single_seeds = [DEFAULT_SEEDS[0]]
            all_jobs = []
            for ds in datasets:
                is_cifar = (ds == 'cifar10')
                seeds = seeds_cifar10 if is_cifar else seeds_mnist

                if 'main' in selected_types:
                    exp_single = 'main_table_cifar10_single' if is_cifar else 'main_table_single'
                    for seed in seeds:
                        for attack in sel_attacks:
                            for method in sel_methods:
                                all_jobs.append(ExperimentRunner(
                                    exp_single, f"{ds}|s{seed}|{method}|{attack}",
                                    extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                                               'MVP_SEED': str(seed)}
                                ))

                if 'ablation' in selected_types:
                    exp_single = 'ablation_cifar10_single' if is_cifar else 'ablation_single'
                    for seed in single_seeds:
                        for attack in ABLATION_ATTACKS:
                            for method in ABLATION_METHODS:
                                all_jobs.append(ExperimentRunner(
                                    exp_single, f"{ds}|abl|s{seed}|{method}|{attack}",
                                    extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                                               'MVP_SEED': str(seed)}
                                ))

                tau_methods_all = ['contra', 'sltrust']
                tau_attacks_all = ['burn_in', 'oscillating', 'sign_flip', 'gaussian',
                                   'slow_poison', 'gradual_flip', 'backdoor']

                if 'tau' in selected_types:
                    exp_single = 'tau_cifar10_single' if is_cifar else 'tau_mnist_single'
                    for seed in single_seeds:
                        for attack in tau_attacks_all:
                            for method in tau_methods_all:
                                all_jobs.append(ExperimentRunner(
                                    exp_single, f"{ds}|tau|s{seed}|{method}|{attack}",
                                    extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                                               'MVP_SEED': str(seed)}
                                ))

                if 'tau_decay' in selected_types:
                    decay_methods = ['contra_hard', 'sltrust_m'] if 'tau' in selected_types else ['contra', 'contra_hard', 'sltrust', 'sltrust_m']
                    exp_single = 'tau_decay_cifar10_single' if is_cifar else 'tau_decay_mnist_single'
                    for seed in single_seeds:
                        for method in decay_methods:
                            all_jobs.append(ExperimentRunner(
                                exp_single, f"{ds}|decay|s{seed}|{method}",
                                extra_env={'MVP_METHOD': method, 'MVP_SEED': str(seed)}
                            ))

            # 分项统计
            from collections import Counter
            job_counts = Counter(j.exp_key for j in all_jobs)
            for k, v in sorted(job_counts.items()):
                console.print(f"  [dim]{k}: {v}[/dim]")
            console.print(f"\n[bold cyan]共 {len(all_jobs)} 个 job，开始并行调度...[/bold cyan]")
            if not run_parallel(all_jobs):
                console.print("[red]存在失败任务，已停止统一出图。[/red]")
                continue

            # 6. 统一出图
            console.print(f"\n[cyan]开始合并出图...[/cyan]")
            for ds in datasets:
                is_cifar = (ds == 'cifar10')
                seeds = seeds_cifar10 if is_cifar else seeds_mnist
                if 'main' in selected_types:
                    exp_plot = 'main_table_cifar10_plot' if is_cifar else 'main_table_plot'
                    for seed in seeds:
                        run_single(exp_plot, f'主表 {ds.upper()} seed={seed} 出图',
                                   {'MVP_SEED': str(seed)})
                if 'ablation' in selected_types:
                    exp_plot = 'ablation_cifar10_plot' if is_cifar else 'ablation_plot'
                    for seed in single_seeds:
                        run_single(exp_plot, f'消融 {ds.upper()} seed={seed} 出图',
                                   {'MVP_SEED': str(seed)})
                if 'tau' in selected_types:
                    exp_plot = 'tau_cifar10' if is_cifar else 'tau_mnist'
                    run_single(exp_plot, f'τ演化 {ds.upper()} 出图',
                               {'MVP_PLOT_ONLY': '1'})
                if 'tau_decay' in selected_types:
                    if 'tau' in selected_types:
                        exp_single = 'tau_decay_cifar10_single' if is_cifar else 'tau_decay_mnist_single'
                        for method in ('contra', 'sltrust'):
                            run_single(exp_single, f'{ds.upper()} τ缓存复用 {method}',
                                       {'MVP_METHOD': method, 'MVP_SEED': str(single_seeds[0])})
                    exp_plot = 'tau_decay_cifar10' if is_cifar else 'tau_decay_mnist'
                    run_single(exp_plot, f'τ衰减 {ds.upper()} 出图')

            console.print()
            console.print(render_results_table())
            continue

        found = False
        for gk, group in EXPERIMENTS.items():
            for idx_key, exp_key, label in group['items']:
                if choice.upper() in (idx_key.upper(), exp_key.lower()):
                    run_single(exp_key, label)
                    found = True
                    break
            if found:
                break

        if not found:
            console.print(f"[red]未知: {choice}[/red]")
            continue

        console.print()
        console.print(render_results_table())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]退出[/dim]")
    except EOFError:
        console.print("\n[yellow]未检测到交互输入，TUI 已退出。[/yellow]")
        console.print("[dim]请在 PowerShell 或 CMD 中运行: python tui.py[/dim]")
