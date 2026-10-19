"""
SLTrust-DFL Experiment Dashboard (TUI) — refactor-v2
Simplified: 5 methods × 6 attacks, 4 menu entries.
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
        f"Expected: {PROJ_DIR / 'utils' / '__init__.py'}"
    ) from exc

DATA_DIR = str(PROJ_DIR / 'data')
console = Console(width=80)

ALL_METHODS = [
    ('fedavg',      'FedAvg'),
    ('sama',        'SAMA'),
    ('contra',      'CONTRA'),
    ('contra_hard', 'CONTRA-Hard'),
    ('sltrust',     'SLTrust'),
]

ALL_ATTACKS = [
    ('none',             'No Attack'),
    ('gaussian',         'Gaussian Noise'),
    ('broken_node',      'Broken Node'),
    ('burn_in',          'Burn-in Gaussian'),
    ('oscillating',      'Oscillating Gaussian'),
    ('gradual_gaussian', 'Gradual Gaussian'),
]


# ──────────────────────────────────────────────────────────
# GPU info
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
        return Panel("[dim]GPU info unavailable[/dim]", title="GPU", border_style="dim")
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
# Dataset check
# ──────────────────────────────────────────────────────────

def check_and_prompt_datasets():
    status = check_all_datasets(DATA_DIR)
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Dataset", style="bold", width=12)
    table.add_column("Status")
    for name, ok in status.items():
        table.add_row(name.upper(), "[green]OK[/green]" if ok else "[red]Missing[/red]")
    all_ok = all(status.values())
    console.print(Panel(table,
                        title="[bold green]Datasets[/bold green]" if all_ok
                        else "[bold yellow]Datasets (missing)[/bold yellow]",
                        border_style="green" if all_ok else "yellow"))
    missing = [name for name, ok in status.items() if not ok]
    if not missing:
        return
    if Confirm.ask("Download missing datasets?", default=True):
        for name in missing:
            console.print(f"\n[cyan]Downloading {name.upper()}...[/cyan]")
            script = (f"import sys; sys.path.insert(0, r'{PROJ_DIR}'); "
                      f"from utils import download_dataset; "
                      f"download_dataset('{name}', r'{DATA_DIR}')")
            subprocess.run([sys.executable, '-u', '-c', script], cwd=str(PROJ_DIR))


# ──────────────────────────────────────────────────────────
# Experiment runner (unchanged core)
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
        log_dir = PROJ_DIR / 'logs' / f"{self.start_time:%Y%m%d}"
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
            return "[green]Done[/green]" if self.returncode == 0 else f"[red]Failed({self.returncode})[/red]"
        if self.running:
            return f"[cyan]Running {self.elapsed()}[/cyan]"
        return "[dim]Waiting[/dim]"


# ──────────────────────────────────────────────────────────
# Single experiment run (with Live output)
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
                    elif 'Round' in line or '---' in line:
                        log_text.append(line + "\n", style="cyan")
                    else:
                        log_text.append(line + "\n", style="dim white")
                layout = Layout()
                layout.split_column(
                    Layout(render_gpu_panel(), size=4),
                    Layout(Panel(log_text,
                                 title=f"[bold cyan]{label}  {job.elapsed()}[/bold cyan]",
                                 border_style="cyan")),
                    Layout(Panel("[dim]Ctrl+C to cancel[/dim]", border_style="dim"), size=3),
                )
                live.update(layout)
                time.sleep(0.25)
    except KeyboardInterrupt:
        job.stop()
        console.print("\n[yellow]Interrupted[/yellow]")
        return

    if job.returncode == 0:
        console.print(Rule(f"[green]{label} done  {job.elapsed()}[/green]"))
    else:
        console.print(Rule(f"[red]{label} failed (exit {job.returncode})[/red]"))
    for line in job.get_last_lines(6):
        console.print(f"  [dim]{line}[/dim]")


# ──────────────────────────────────────────────────────────
# Parallel run
# ──────────────────────────────────────────────────────────

def run_parallel(jobs: list, max_concurrent: int = 6):
    unique_jobs = []
    seen = set()
    for job in jobs:
        identity = (job.exp_key, tuple(sorted(job.extra_env.items())))
        if identity in seen:
            continue
        seen.add(identity)
        unique_jobs.append(job)
    if len(unique_jobs) != len(jobs):
        console.print(f"[yellow]Dedup: skipped {len(jobs) - len(unique_jobs)} duplicate jobs[/yellow]")
    jobs = unique_jobs
    pending = list(jobs)
    started = []

    def _fill_slots():
        while pending and sum(1 for j in started if j.running) < max_concurrent:
            j = pending.pop(0)
            j.start()
            started.append(j)

    _fill_slots()

    interrupted = False
    total_jobs = len(jobs)
    eta_str = ""
    last_done_count = 0
    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                _fill_slots()
                done_jobs = [j for j in started if j.done]
                running_jobs = [j for j in started if j.running]

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
                        if 'acc=' in line.lower() or '%' in line:
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

                if done_jobs or pending:
                    table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
                    table.add_column("Job", style="white")
                    table.add_column("Status", width=18)
                    for job in done_jobs:
                        table.add_row(job.label, job.status_str())
                    for job in pending:
                        table.add_row(job.label, "[dim]Queued[/dim]")
                    renderables.append(Panel(table, border_style="dim", padding=(0, 1)))

                slots_text = f"Running: {len(running_jobs)}  Done: {len(done_jobs)}/{total_jobs}  Queue: {len(pending)}{eta_str}"
                jobs_panel = Panel(
                    Group(*renderables) if renderables else Text(""),
                    title=f"[bold cyan]Progress  {slots_text}[/bold cyan]",
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
        console.print("\n[yellow]Interrupted[/yellow]")

    if not interrupted:
        failed = [job for job in started if job.returncode != 0]
        style = "bold red" if failed else "bold green"
        title = f"{len(failed)} jobs failed" if failed else "All done"
        console.print(Rule(f"[{style}]{title}[/{style}]"))
        for job in started:
            mark = "[green]OK[/green]" if job.returncode == 0 else "[red]FAIL[/red]"
            console.print(f"  {mark} {job.label}  {job.elapsed()}")
        return not failed
    return False


# ──────────────────────────────────────────────────────────
# Selection helpers
# ──────────────────────────────────────────────────────────

def _parse_selection(raw, entries):
    raw = raw.strip().lower()
    if raw in ('', 'a', 'all'):
        return [key for key, _ in entries]
    selected = []
    for token in raw.replace(',', ' ').split():
        if token.isdigit() and 1 <= int(token) <= len(entries):
            selected.append(entries[int(token) - 1][0])
        elif token in {key for key, _ in entries}:
            selected.append(token)
    return list(dict.fromkeys(selected))


def choose_multi(title, entries):
    console.print(f'\n[bold]{title}[/bold] (space/comma separated; blank = all)')
    for idx, (key, label) in enumerate(entries, 1):
        console.print(f'  [cyan]{idx}[/cyan] {label} [dim]({key})[/dim]')
    selected = _parse_selection(Prompt.ask('Select', default=''), entries)
    if not selected:
        selected = [key for key, _ in entries]
    console.print(f'[green]Selected: {", ".join(selected)}[/green]')
    return selected


def choose_seeds():
    from config_loader import load_experiment_config
    base_seed = int(load_experiment_config('main_table_mnist')['experiment']['seed'])
    configured = [base_seed, 123, 456, 789, 1024]
    raw = Prompt.ask("[bold]Number of seeds[/bold] (1=quick, 3-5=paper)", default="1").strip()
    try:
        n = max(1, min(10, int(raw)))
    except ValueError:
        n = 1
    seeds = configured[:n]
    console.print(f'[green]{n} seed(s): {seeds}[/green]')
    return seeds


def choose_dataset():
    console.print('\n[bold]Dataset[/bold]')
    console.print('  [cyan]1[/cyan] MNIST    [cyan]2[/cyan] CIFAR-10    [cyan]3[/cyan] Both')
    raw = Prompt.ask('Select', default='1').strip()
    if raw == '2':
        return ['cifar10']
    if raw == '3':
        return ['mnist', 'cifar10']
    return ['mnist']


# ──────────────────────────────────────────────────────────
# Menu actions
# ──────────────────────────────────────────────────────────

def action_main_table():
    datasets = choose_dataset()
    methods = choose_multi('Methods', ALL_METHODS)
    attacks = choose_multi('Attacks', ALL_ATTACKS)
    seeds = choose_seeds()
    jobs = []
    for ds in datasets:
        exp_single = 'main_table_cifar10_single' if ds == 'cifar10' else 'main_table_single'
        for seed in seeds:
            for attack in attacks:
                for method in methods:
                    jobs.append(ExperimentRunner(
                        exp_single, f"{ds}|s{seed}|{method}|{attack}",
                        extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                                   'MVP_SEED': str(seed)}
                    ))
    console.print(f'\n[bold cyan]{len(jobs)} jobs total[/bold cyan]')
    if not run_parallel(jobs):
        return
    for ds in datasets:
        exp_plot = 'main_table_cifar10_plot' if ds == 'cifar10' else 'main_table_plot'
        for seed in seeds:
            run_single(exp_plot, f'Plot {ds.upper()} seed={seed}', {'MVP_SEED': str(seed)})


def action_ablation():
    ABLATION_METHODS = ['sltrust', 'sltrust_no_ratio', 'sltrust_no_clip']
    ABLATION_ATTACKS = ['burn_in', 'oscillating']
    datasets = choose_dataset()
    seeds = choose_seeds()
    jobs = []
    for ds in datasets:
        exp_single = 'ablation_cifar10_single' if ds == 'cifar10' else 'ablation_single'
        for seed in seeds:
            for attack in ABLATION_ATTACKS:
                for method in ABLATION_METHODS:
                    jobs.append(ExperimentRunner(
                        exp_single, f"{ds}|abl|s{seed}|{method}|{attack}",
                        extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                                   'MVP_SEED': str(seed)}
                    ))
    console.print(f'\n[bold cyan]{len(jobs)} jobs total[/bold cyan]')
    if not run_parallel(jobs):
        return
    for ds in datasets:
        exp_plot = 'ablation_cifar10_plot' if ds == 'cifar10' else 'ablation_plot'
        for seed in seeds:
            run_single(exp_plot, f'Ablation plot {ds.upper()} seed={seed}', {'MVP_SEED': str(seed)})


def action_tau():
    TAU_METHODS = ['contra', 'sltrust']
    TAU_ATTACKS = [k for k, _ in ALL_ATTACKS if k != 'none']
    datasets = choose_dataset()
    jobs = []
    for ds in datasets:
        exp_single = 'tau_cifar10_single' if ds == 'cifar10' else 'tau_mnist_single'
        seed = 42
        for attack in TAU_ATTACKS:
            for method in TAU_METHODS:
                jobs.append(ExperimentRunner(
                    exp_single, f"{ds}|tau|{method}|{attack}",
                    extra_env={'MVP_METHOD': method, 'MVP_ATTACK': attack,
                               'MVP_SEED': str(seed)}
                ))
    console.print(f'\n[bold cyan]{len(jobs)} jobs total[/bold cyan]')
    if not run_parallel(jobs):
        return
    for ds in datasets:
        exp_plot = 'tau_cifar10' if ds == 'cifar10' else 'tau_mnist'
        run_single(exp_plot, f'Tau plot {ds.upper()}', {'MVP_PLOT_ONLY': '1'})


def action_sensitivity():
    from config_loader import load_experiment_config
    from experiment_matrix import sensitivity_tasks
    cfg = load_experiment_config('sensitivity_mnist')
    tasks = sensitivity_tasks(cfg)
    jobs = []
    for task in tasks:
        env = {'MVP_METHOD': task['method'], 'MVP_ATTACK': task['attack'],
               'MVP_SEED': str(task['seed'])}
        if task['parameter'] == 'warmup':
            env['MVP_WARMUP'] = str(task['value'])
        elif task['parameter'] == 'period':
            env['MVP_PERIOD'] = str(task['value'])
        detail = f"{task['method']}|{task['attack']}|{task['parameter']}={task['value']}"
        jobs.append(ExperimentRunner('sensitivity_single', detail, env))
    console.print(f'\n[bold cyan]{len(jobs)} sensitivity jobs[/bold cyan]')
    if not run_parallel(jobs):
        return
    run_single('sensitivity_plot', 'Sensitivity plots')


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    if not sys.stdin.isatty():
        console.print('[red]tui.py requires an interactive terminal.[/red]')
        return
    console.clear()
    console.print(render_gpu_panel())
    check_and_prompt_datasets()

    while True:
        console.print('\n[bold white]SLTrust-DFL Experiment Dashboard[/bold white]')
        console.print('  [cyan]1[/cyan] Main table  (5 methods x 6 attacks)')
        console.print('  [cyan]2[/cyan] Ablation    (Full / NoRatio / NoClip)')
        console.print('  [cyan]3[/cyan] Tau evolution (CONTRA vs SLTrust)')
        console.print('  [cyan]4[/cyan] Sensitivity  (warmup / period sweep)')
        console.print('  [cyan]q[/cyan] Quit')
        choice = Prompt.ask('Select', default='1').strip().lower()

        if choice in ('q', 'quit', 'exit'):
            console.print('[dim]Bye.[/dim]')
            break
        elif choice == '1':
            action_main_table()
        elif choice == '2':
            action_ablation()
        elif choice == '3':
            action_tau()
        elif choice == '4':
            action_sensitivity()
        else:
            console.print(f'[red]Unknown: {choice}[/red]')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Bye.[/dim]")
    except EOFError:
        console.print("\n[yellow]No interactive input, exiting.[/yellow]")
