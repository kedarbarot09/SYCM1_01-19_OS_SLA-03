"""
CPU Scheduling Simulator (Python / Tkinter)

Algorithms
----------
1. Priority Scheduling      - non-preemptive or preemptive, user-supplied priority
2. Multilevel Queue (MLQ)   - System (Priority) -> Interactive (Round Robin) -> Batch (FCFS)
3. ASH-PLR                  - Adaptive Scheduling Heuristic with Priority Linear Regression
                              (Malabanan, Foryasen, Balay-odao & Tandingan Jr., 2025)

ASH-PLR idea (as used here)
---------------------------
Instead of trusting only the fixed priority supplied with each process, a linear
regression model PREDICTS a suitable priority from process characteristics:

    predicted_priority = b0 + b1*burst + b2*arrival + b3*given_priority

The predicted priority (lower number = higher priority) is then used by a priority
scheduler. Metrics reported, as in the paper: average turnaround time, average
waiting time and number of context switches.

NOTE: The paper's exact features / training data are not reproduced here. The model
is trained on a seeded synthetic workload (see ASHPLRModel). To match the paper more
closely, edit LABEL_WEIGHTS or replace _make_training_data() with the paper's data.
"""

import random
import tkinter as tk
from tkinter import ttk, messagebox

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
QUEUES = ["System", "Interactive", "Batch"]
ALGORITHMS = ["Priority Scheduling", "Multilevel Queue", "ASH-PLR", "Compare All"]
PRIORITY_MIN, PRIORITY_MAX = 1, 10

SAMPLE_PROCESSES = [
    ("P1", 0, 5, 1, "System"),
    ("P2", 1, 3, 3, "Interactive"),
    ("P3", 2, 8, 4, "Batch"),
    ("P4", 3, 2, 2, "System"),
    ("P5", 4, 4, 5, "Batch"),
]

GANTT_COLORS = [
    "#4E79A7", "#E8833A", "#3F9B5A", "#D6494B", "#5BA3A0",
    "#B8860B", "#8E6BB0", "#C06C84", "#8C6D4F", "#5F6B7A",
]


# ----------------------------------------------------------------------
# Linear regression (pure Python, no external libraries needed)
# ----------------------------------------------------------------------
def solve_linear_system(A, b):
    """Solve A x = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        known = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - known) / M[i][i]
    return x


class ASHPLRModel:
    """Linear regression model that predicts a priority level (1 = highest)."""

    # Weights used to build the "ideal priority" label of the synthetic training
    # set: shorter bursts -> higher priority (consistent with the paper's finding
    # that ASH-PLR works well for short-burst processes).
    LABEL_WEIGHTS = {"burst": 0.65, "arrival": 0.10, "priority": 0.25}

    def __init__(self, samples=300, seed=42):
        self.samples = samples
        X, y = self._make_training_data(samples, seed)
        self.coef = self._fit(X, y)        # [b0, b_burst, b_arrival, b_priority]
        self.r2 = self._r_squared(X, y)

    def _make_training_data(self, n, seed):
        rng = random.Random(seed)
        w = self.LABEL_WEIGHTS
        X, y = [], []
        for _ in range(n):
            burst = rng.randint(1, 20)
            arrival = rng.randint(0, 20)
            priority = rng.randint(PRIORITY_MIN, PRIORITY_MAX)
            score = (
                w["burst"] * (burst - 1) / 19
                + w["arrival"] * arrival / 20
                + w["priority"] * (priority - 1) / 9
            )
            label = 1 + 9 * score + rng.gauss(0, 0.3)
            label = min(PRIORITY_MAX, max(PRIORITY_MIN, label))
            X.append([1.0, burst, arrival, priority])   # leading 1.0 = intercept
            y.append(label)
        return X, y

    @staticmethod
    def _fit(X, y):
        """Least squares via the normal equations: (X^T X) b = X^T y."""
        k = len(X[0])
        A = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
        b = [sum(r[i] * t for r, t in zip(X, y)) for i in range(k)]
        for i in range(k):
            A[i][i] += 1e-6          # tiny ridge term for numerical stability
        return solve_linear_system(A, b)

    def _raw(self, row):
        return sum(c * v for c, v in zip(self.coef, row))

    def _r_squared(self, X, y):
        mean = sum(y) / len(y)
        ss_tot = sum((t - mean) ** 2 for t in y)
        ss_res = sum((t - self._raw(r)) ** 2 for r, t in zip(X, y))
        return 1 - ss_res / ss_tot

    def predict(self, burst, arrival, priority):
        raw = self._raw([1.0, burst, arrival, priority])
        return round(min(PRIORITY_MAX, max(PRIORITY_MIN, raw)), 2)

    def equation(self):
        b0, b1, b2, b3 = self.coef
        return (
            f"Predicted Priority = {b0:.3f} {b1:+.3f} x Burst "
            f"{b2:+.3f} x Arrival {b3:+.3f} x Given Priority"
        )


# ----------------------------------------------------------------------
# Scheduling engine (no GUI code in here)
# A timeline is a list of (pid, start, end); idle time uses pid "IDLE".
# ----------------------------------------------------------------------
def simulate_priority(procs, key_fn, preemptive):
    """
    Generic priority scheduler. key_fn(p, time, remaining) returns a sort key;
    the SMALLEST key runs next. Non-preemptive runs a process to completion,
    preemptive re-decides whenever a new process arrives.
    """
    remaining = {p["id"]: p["burst"] for p in procs}
    timeline = []
    time = 0

    def add(pid, start, end, merge):
        if merge and timeline and timeline[-1][0] == pid and timeline[-1][2] == start:
            timeline[-1] = (pid, timeline[-1][1], end)
        else:
            timeline.append((pid, start, end))

    while any(remaining.values()):
        available = [
            p for p in procs
            if p["arrival"] <= time and remaining[p["id"]] > 0
        ]

        if not available:
            next_arrival = min(
                p["arrival"] for p in procs if remaining[p["id"]] > 0
            )
            add("IDLE", time, next_arrival, False)
            time = next_arrival
            continue

        chosen = min(
            available, key=lambda p: key_fn(p, time, remaining[p["id"]])
        )
        run = remaining[chosen["id"]]
        if preemptive:
            # Re-decide only when a new process arrives (a possible preemption point)
            upcoming = [
                p["arrival"] for p in procs
                if p["arrival"] > time and remaining[p["id"]] > 0
            ]
            if upcoming:
                run = min(run, min(upcoming) - time)
        add(chosen["id"], time, time + run, True)
        time += run
        remaining[chosen["id"]] -= run

    return timeline


def key_manual(p, time, remaining):
    """Plain priority scheduling: smaller priority number wins."""
    return (p["priority"], p["arrival"], p["order"])


def make_ashplr_key(predicted, aging):
    """
    ASH-PLR key: regression-predicted priority, adjusted by an aging term so that
    long-waiting processes gradually gain priority (anti-starvation, adaptive part).
    Set aging = 0 for pure regression-predicted priority.
    """
    def key(p, time, remaining):
        waited = time - p["arrival"] - (p["burst"] - remaining)
        return (predicted[p["id"]] - aging * waited, p["arrival"], p["order"])
    return key


def simulate_mlq(procs, quantum):
    """
    Multilevel Queue with fixed precedence System -> Interactive -> Batch.
      System      : priority scheduling (non-preemptive)
      Interactive : Round Robin with the given time quantum
      Batch       : FCFS (non-preemptive)
    The highest-precedence queue with a ready process is chosen whenever the CPU
    becomes free.
    """
    remaining = {p["id"]: p["burst"] for p in procs}
    system = [p for p in procs if p["queue"] == "System"]
    interactive = sorted(
        [p for p in procs if p["queue"] == "Interactive"],
        key=lambda p: (p["arrival"], p["order"]),
    )
    batch = [p for p in procs if p["queue"] == "Batch"]

    ready_rr = []
    admitted = set()
    done = set()
    timeline = []
    time = 0

    def admit():
        for p in interactive:
            if p["arrival"] <= time and p["id"] not in admitted:
                ready_rr.append(p)
                admitted.add(p["id"])

    while len(done) < len(procs):
        admit()

        # 1) System queue: priority scheduling, runs to completion
        avail_sys = [
            p for p in system if p["arrival"] <= time and p["id"] not in done
        ]
        if avail_sys:
            sel = min(avail_sys, key=lambda p: (p["priority"], p["arrival"], p["order"]))
            start = time
            time += remaining[sel["id"]]
            remaining[sel["id"]] = 0
            done.add(sel["id"])
            timeline.append((sel["id"], start, time))
            continue

        # 2) Interactive queue: Round Robin
        if ready_rr:
            sel = ready_rr.pop(0)
            run = min(quantum, remaining[sel["id"]])
            start = time
            time += run
            remaining[sel["id"]] -= run
            timeline.append((sel["id"], start, time))
            admit()   # processes that arrived during the slice queue up first
            if remaining[sel["id"]] == 0:
                done.add(sel["id"])
            else:
                ready_rr.append(sel)
            continue

        # 3) Batch queue: FCFS, runs to completion
        avail_batch = [
            p for p in batch if p["arrival"] <= time and p["id"] not in done
        ]
        if avail_batch:
            sel = min(avail_batch, key=lambda p: (p["arrival"], p["order"]))
            start = time
            time += remaining[sel["id"]]
            remaining[sel["id"]] = 0
            done.add(sel["id"])
            timeline.append((sel["id"], start, time))
            continue

        # Nothing ready: jump to the next arrival
        next_arrival = min(p["arrival"] for p in procs if p["id"] not in done)
        if next_arrival > time:
            timeline.append(("IDLE", time, next_arrival))
        time = next_arrival

    return timeline


def compute_metrics(procs, timeline):
    """Per-process CT/TAT/WT plus the overall performance parameters."""
    completion = {}
    for pid, start, end in timeline:
        if pid != "IDLE":
            completion[pid] = end

    results = []
    for p in procs:
        ct = completion[p["id"]]
        tat = ct - p["arrival"]
        wt = tat - p["burst"]
        results.append({"id": p["id"], "completion": ct, "turnaround": tat, "waiting": wt})

    n = len(results)
    first_arrival = min(p["arrival"] for p in procs)
    elapsed = max(completion.values()) - first_arrival
    total_burst = sum(p["burst"] for p in procs)

    # Context switch = CPU moves directly from one process to a different one
    switches, prev = 0, None
    for pid, start, end in timeline:
        if pid == "IDLE":
            prev = None
            continue
        if prev is not None and prev != pid:
            switches += 1
        prev = pid

    summary = {
        "avg_wt": sum(r["waiting"] for r in results) / n,
        "avg_tat": sum(r["turnaround"] for r in results) / n,
        "util": total_burst / elapsed * 100 if elapsed > 0 else 0,
        "throughput": n / elapsed if elapsed > 0 else 0,
        "switches": switches,
    }
    return results, summary


def parse_int(text, label):
    try:
        return int(text.strip())
    except ValueError:
        raise ValueError(f"{label} must be a whole number.")


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class SchedulerApp:
    def __init__(self, root):
        self.root = root
        self.processes = []
        self.runs = {}                # algorithm name -> last run record
        self.model = ASHPLRModel()
        self._build_ui()
        self.refresh_predictions()

    # ---------------- UI construction ----------------
    def _build_ui(self):
        root = self.root
        root.title("CPU Scheduling Simulator")
        root.geometry("1300x880")
        root.minsize(1150, 780)
        root.configure(bg="#F2F5F7")

        style = ttk.Style(root)
        style.configure(".", font=("Arial", 11))
        style.configure("Treeview", font=("Arial", 11), rowheight=26)
        style.configure("Treeview.Heading", font=("Arial", 11, "bold"))
        style.configure("TLabelframe.Label", font=("Arial", 11, "bold"))
        style.configure("TNotebook.Tab", font=("Arial", 11, "bold"), padding=(14, 6))

        tk.Label(
            root, text="CPU SCHEDULING SIMULATOR",
            font=("Arial", 24, "bold"), bg="#183B56", fg="white", pady=14,
        ).pack(fill="x")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=15, pady=10)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 12))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self._build_input_panel(left)
        self._build_process_table(left)
        self._build_controls(left)
        self._build_right_side(right)

    def _build_input_panel(self, parent):
        form = ttk.LabelFrame(parent, text="Process Details", padding=12)
        form.pack(fill="x", pady=(0, 10))

        names = ["Process ID", "Arrival Time", "Burst Time", "Priority"]
        entries = []
        for row, text in enumerate(names):
            ttk.Label(form, text=text).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            entry = ttk.Entry(form, width=20)
            entry.grid(row=row, column=1, padx=6, pady=4)
            entry.bind("<Return>", lambda event: self.add_process())
            entries.append(entry)
        (self.id_entry, self.arrival_entry,
         self.burst_entry, self.priority_entry) = entries

        ttk.Label(form, text="Queue Type").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        self.queue_box = ttk.Combobox(form, values=QUEUES, state="readonly", width=18)
        self.queue_box.grid(row=4, column=1, padx=6, pady=4)
        self.queue_box.set("System")

        ttk.Button(form, text="Add Process", command=self.add_process).grid(
            row=5, column=0, columnspan=2, pady=(8, 2)
        )
        ttk.Label(
            form, text="Smaller priority number = higher priority.\n"
                       "Queue Type is used by Multilevel Queue only.",
            foreground="#555555", font=("Arial", 9),
        ).grid(row=6, column=0, columnspan=2, pady=(4, 0))

    def _build_process_table(self, parent):
        frame = ttk.LabelFrame(parent, text="Process Table", padding=10)
        frame.pack(fill="both", expand=True, pady=(0, 10))

        holder = ttk.Frame(frame)
        holder.pack(fill="both", expand=True)

        columns = ("ID", "Arrival", "Burst", "Priority", "Queue")
        self.process_table = ttk.Treeview(
            holder, columns=columns, show="headings", height=8, selectmode="extended"
        )
        widths = (60, 65, 60, 70, 95)
        for col, width in zip(columns, widths):
            self.process_table.heading(col, text=col)
            self.process_table.column(col, anchor="center", width=width)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self.process_table.yview)
        self.process_table.configure(yscrollcommand=scroll.set)
        self.process_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Remove Selected", command=self.remove_selected).pack(side="left", padx=2)
        ttk.Button(buttons, text="Load Sample", command=self.load_sample).pack(side="left", padx=2)
        ttk.Button(buttons, text="Clear All", command=self.clear_all).pack(side="left", padx=2)

    def _build_controls(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Simulation Controls", padding=12)
        ctrl.pack(fill="x")

        ttk.Label(ctrl, text="Algorithm").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.algorithm_box = ttk.Combobox(ctrl, values=ALGORITHMS, state="readonly", width=20)
        self.algorithm_box.grid(row=0, column=1, padx=6, pady=4)
        self.algorithm_box.set("ASH-PLR")

        self.preemptive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctrl, text="Preemptive (Priority / ASH-PLR)", variable=self.preemptive_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        ttk.Label(ctrl, text="Time Quantum (MLQ)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.quantum_var = tk.StringVar(value="2")
        ttk.Spinbox(ctrl, from_=1, to=50, width=8, textvariable=self.quantum_var).grid(
            row=2, column=1, sticky="w", padx=6, pady=4
        )

        ttk.Label(ctrl, text="Aging Factor (ASH-PLR)").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.aging_var = tk.StringVar(value="0.25")
        ttk.Spinbox(
            ctrl, from_=0.0, to=5.0, increment=0.05, width=8, textvariable=self.aging_var
        ).grid(row=3, column=1, sticky="w", padx=6, pady=4)

        ttk.Button(ctrl, text="Run Simulation", command=self.run_simulation).grid(
            row=4, column=0, columnspan=2, pady=(10, 0)
        )

    def _build_right_side(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self.tab_results = ttk.Frame(self.notebook, padding=10)
        self.tab_compare = ttk.Frame(self.notebook, padding=10)
        self.tab_model = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_results, text="Results")
        self.notebook.add(self.tab_compare, text="Comparison")
        self.notebook.add(self.tab_model, text="ASH-PLR Model")

        # ---- Results tab ----
        res_frame = ttk.LabelFrame(self.tab_results, text="Simulation Results", padding=8)
        res_frame.pack(fill="x")
        columns = ("ID", "Priority / Queue Used", "Completion Time", "Turnaround Time", "Waiting Time")
        self.result_table = ttk.Treeview(res_frame, columns=columns, show="headings", height=6)
        for col in columns:
            self.result_table.heading(col, text=col)
            self.result_table.column(col, anchor="center", width=150)
        self.result_table.pack(fill="x")

        gantt_frame = ttk.LabelFrame(self.tab_results, text="CPU Timeline (Gantt Chart)", padding=8)
        gantt_frame.pack(fill="x", pady=10)
        self.gantt_canvas = tk.Canvas(gantt_frame, height=105, bg="white", highlightthickness=0)
        gantt_scroll = ttk.Scrollbar(gantt_frame, orient="horizontal", command=self.gantt_canvas.xview)
        self.gantt_canvas.configure(xscrollcommand=gantt_scroll.set)
        self.gantt_canvas.pack(fill="x")
        gantt_scroll.pack(fill="x")

        sum_frame = ttk.LabelFrame(self.tab_results, text="Performance Summary", padding=8)
        sum_frame.pack(fill="both", expand=True)
        self.summary_text = tk.Text(
            sum_frame, height=9, wrap="word", font=("Arial", 11),
            relief="flat", bg="#F2F5F7", state="disabled",
        )
        self.summary_text.pack(fill="both", expand=True)
        self._set_text(self.summary_text, "Enter processes and run a simulation.")

        # ---- Comparison tab ----
        ttk.Label(
            self.tab_compare,
            text="Results of the algorithms run on the current processes. "
                 "Choose 'Compare All' to run all three at once.",
            wraplength=720,
        ).pack(anchor="w", pady=(0, 8))
        cmp_columns = ("Algorithm", "Avg Waiting", "Avg Turnaround",
                       "CPU Util %", "Throughput", "Context Switches")
        self.compare_table = ttk.Treeview(
            self.tab_compare, columns=cmp_columns, show="headings", height=5
        )
        for col in cmp_columns:
            self.compare_table.heading(col, text=col)
            self.compare_table.column(col, anchor="center", width=110)
        self.compare_table.column("Algorithm", width=270, anchor="w")
        self.compare_table.pack(fill="x")
        self.compare_text = tk.Text(
            self.tab_compare, height=8, wrap="word", font=("Arial", 11),
            relief="flat", bg="#F2F5F7", state="disabled",
        )
        self.compare_text.pack(fill="both", expand=True, pady=(10, 0))

        # ---- Model tab ----
        self.model_equation = ttk.Label(
            self.tab_model, text="", font=("Arial", 12, "bold"), wraplength=720
        )
        self.model_equation.pack(anchor="w")
        self.model_info = ttk.Label(self.tab_model, text="", wraplength=720, foreground="#444444")
        self.model_info.pack(anchor="w", pady=(4, 10))

        pred_columns = ("ID", "Burst", "Arrival", "Given Priority", "Predicted Priority", "Rank")
        self.pred_table = ttk.Treeview(
            self.tab_model, columns=pred_columns, show="headings", height=8
        )
        for col in pred_columns:
            self.pred_table.heading(col, text=col)
            self.pred_table.column(col, anchor="center", width=110)
        self.pred_table.pack(fill="x")

        self.model_notes = tk.Text(
            self.tab_model, height=9, wrap="word", font=("Arial", 11),
            relief="flat", bg="#F2F5F7", state="disabled",
        )
        self.model_notes.pack(fill="both", expand=True, pady=(10, 0))
        self._set_text(
            self.model_notes,
            "How ASH-PLR works in this simulator\n"
            "1. A linear regression model predicts a priority level (1 = highest, "
            "10 = lowest) for every process from its burst time, arrival time and "
            "given priority.\n"
            "2. The predicted priority replaces the fixed priority in a priority "
            "scheduler (non-preemptive or preemptive).\n"
            "3. Adaptive step: while a process waits, the Aging Factor lowers its "
            "effective priority number, so long-waiting processes cannot starve. "
            "Set the factor to 0 for pure regression-predicted priority.\n"
            "4. The model is trained on a seeded synthetic workload in which "
            "shorter bursts are given higher priority; it is a stand-in for the "
            "paper's training data.",
        )

        self.model_equation.config(text=self.model.equation())
        self.model_info.config(
            text=f"R-squared on training data: {self.model.r2:.3f}   |   "
                 f"Training samples: {self.model.samples}   |   "
                 f"Prediction is clamped to {PRIORITY_MIN}-{PRIORITY_MAX}"
        )

    # ---------------- helpers ----------------
    @staticmethod
    def _set_text(widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def reset_outputs(self):
        """Clear every result view (called whenever the process list changes)."""
        self.runs = {}
        for table in (self.result_table, self.compare_table):
            for item in table.get_children():
                table.delete(item)
        self.gantt_canvas.delete("all")
        self._set_text(self.summary_text, "Enter processes and run a simulation.")
        self._set_text(self.compare_text, "")

    def refresh_predictions(self):
        for item in self.pred_table.get_children():
            self.pred_table.delete(item)

        predicted = {
            p["id"]: self.model.predict(p["burst"], p["arrival"], p["priority"])
            for p in self.processes
        }
        ranked = sorted(
            self.processes,
            key=lambda p: (predicted[p["id"]], p["arrival"]),
        )
        rank = {p["id"]: i + 1 for i, p in enumerate(ranked)}

        for p in self.processes:
            self.pred_table.insert(
                "", "end",
                values=(p["id"], p["burst"], p["arrival"], p["priority"],
                        f'{predicted[p["id"]]:.2f}', rank[p["id"]]),
            )

    def _processes_changed(self):
        self.reset_outputs()
        self.refresh_predictions()

    # ---------------- process list actions ----------------
    def add_process(self):
        try:
            pid = self.id_entry.get().strip()
            if not pid:
                raise ValueError("Enter a Process ID.")
            if pid.upper() == "IDLE":
                raise ValueError("'IDLE' is reserved. Choose another Process ID.")
            if any(p["id"] == pid for p in self.processes):
                raise ValueError("Process ID must be unique.")

            arrival = parse_int(self.arrival_entry.get(), "Arrival time")
            burst = parse_int(self.burst_entry.get(), "Burst time")
            priority = parse_int(self.priority_entry.get(), "Priority")
            queue = self.queue_box.get()

            if arrival < 0 or burst <= 0:
                raise ValueError("Arrival time must be >= 0 and burst time > 0.")
            if priority < 1:
                raise ValueError("Priority must be at least 1.")
            if not queue:
                raise ValueError("Select a queue type.")
        except ValueError as error:
            messagebox.showerror("Invalid Input", str(error))
            return

        self.processes.append(
            {"id": pid, "arrival": arrival, "burst": burst,
             "priority": priority, "queue": queue}
        )
        self.process_table.insert(
            "", "end", iid=pid, values=(pid, arrival, burst, priority, queue)
        )
        for entry in (self.id_entry, self.arrival_entry,
                      self.burst_entry, self.priority_entry):
            entry.delete(0, tk.END)
        self.id_entry.focus_set()
        self._processes_changed()

    def remove_selected(self):
        selected = self.process_table.selection()
        if not selected:
            messagebox.showinfo("Remove Process", "Select one or more rows to remove.")
            return
        for iid in selected:
            self.process_table.delete(iid)
        self.processes = [p for p in self.processes if p["id"] not in selected]
        self._processes_changed()

    def load_sample(self):
        self._clear_processes()
        for pid, arrival, burst, priority, queue in SAMPLE_PROCESSES:
            self.processes.append(
                {"id": pid, "arrival": arrival, "burst": burst,
                 "priority": priority, "queue": queue}
            )
            self.process_table.insert(
                "", "end", iid=pid, values=(pid, arrival, burst, priority, queue)
            )
        self._processes_changed()

    def _clear_processes(self):
        self.processes.clear()
        for item in self.process_table.get_children():
            self.process_table.delete(item)

    def clear_all(self):
        self._clear_processes()
        self._processes_changed()

    # ---------------- simulation ----------------
    def execute(self, name, quantum, preemptive, aging):
        procs = [dict(p, order=i) for i, p in enumerate(self.processes)]
        mode = "Preemptive" if preemptive else "Non-preemptive"
        extra = ""

        if name == "Priority Scheduling":
            timeline = simulate_priority(procs, key_manual, preemptive)
            used = {p["id"]: str(p["priority"]) for p in procs}
            label = f"Priority Scheduling ({mode})"

        elif name == "Multilevel Queue":
            timeline = simulate_mlq(procs, quantum)
            used = {p["id"]: p["queue"] for p in procs}
            label = f"Multilevel Queue (RR quantum = {quantum})"
            extra = ("Queue Order: System -> Interactive -> Batch\n"
                     f"Interactive Time Quantum: {quantum}\n")

        else:  # ASH-PLR
            predicted = {
                p["id"]: self.model.predict(p["burst"], p["arrival"], p["priority"])
                for p in procs
            }
            timeline = simulate_priority(procs, make_ashplr_key(predicted, aging), preemptive)
            used = {pid: f"{value:.2f}" for pid, value in predicted.items()}
            label = f"ASH-PLR ({mode}, aging = {aging:g})"
            listing = ", ".join(f"{pid} = {value:.2f}" for pid, value in predicted.items())
            extra = f"Predicted Priorities: {listing}\n"

        results, summary = compute_metrics(procs, timeline)
        return {"name": name, "label": label, "timeline": timeline,
                "results": results, "summary": summary, "used": used, "extra": extra}

    def run_simulation(self):
        if not self.processes:
            messagebox.showwarning("No Processes", "Please add at least one process.")
            return

        try:
            quantum = parse_int(self.quantum_var.get(), "Time quantum")
            if quantum < 1:
                raise ValueError("Time quantum must be at least 1.")
            try:
                aging = float(self.aging_var.get())
            except ValueError:
                raise ValueError("Aging factor must be a number.")
            if aging < 0:
                raise ValueError("Aging factor cannot be negative.")
        except ValueError as error:
            messagebox.showerror("Invalid Setting", str(error))
            return

        choice = self.algorithm_box.get()
        names = ALGORITHMS[:3] if choice == "Compare All" else [choice]
        preemptive = self.preemptive_var.get()

        for name in names:
            self.runs[name] = self.execute(name, quantum, preemptive, aging)

        self.show_run(self.runs[names[-1]])
        self.update_comparison()
        self.notebook.select(self.tab_compare if choice == "Compare All" else self.tab_results)

    # ---------------- output display ----------------
    def show_run(self, run):
        for item in self.result_table.get_children():
            self.result_table.delete(item)
        for r in run["results"]:
            self.result_table.insert(
                "", "end",
                values=(r["id"], run["used"][r["id"]],
                        r["completion"], r["turnaround"], r["waiting"]),
            )

        self.draw_gantt(run)

        s = run["summary"]
        order = " -> ".join(f"{pid} ({a}-{b})" for pid, a, b in run["timeline"])
        text = (
            f"Algorithm: {run['label']}\n"
            f"{run['extra']}"
            f"Execution Order: {order}\n"
            f"Average Waiting Time: {s['avg_wt']:.2f}\n"
            f"Average Turnaround Time: {s['avg_tat']:.2f}\n"
            f"Context Switches: {s['switches']}\n"
            f"CPU Utilization: {s['util']:.2f}%\n"
            f"Throughput: {s['throughput']:.3f} processes/unit time"
        )
        self._set_text(self.summary_text, text)

    def draw_gantt(self, run):
        canvas = self.gantt_canvas
        canvas.delete("all")
        timeline = run["timeline"]
        total = timeline[-1][2]

        colors = {
            p["id"]: GANTT_COLORS[i % len(GANTT_COLORS)]
            for i, p in enumerate(self.processes)
        }
        unit = max(16, min(48, 760 // max(total, 1)))
        pad, top, bottom = 15, 15, 65

        for pid, start, end in timeline:
            x0, x1 = pad + start * unit, pad + end * unit
            idle = pid == "IDLE"
            canvas.create_rectangle(
                x0, top, x1, bottom,
                fill="#D9D9D9" if idle else colors[pid], outline="white", width=2,
            )
            canvas.create_text(
                (x0 + x1) / 2, (top + bottom) / 2, text=pid,
                font=("Arial", 10, "bold"), fill="#555555" if idle else "white",
            )
            canvas.create_text(x0, bottom + 14, text=str(start), font=("Arial", 9))
        canvas.create_text(pad + total * unit, bottom + 14, text=str(total), font=("Arial", 9))
        canvas.configure(scrollregion=(0, 0, pad * 2 + total * unit, 95))
        canvas.xview_moveto(0)

    def update_comparison(self):
        for item in self.compare_table.get_children():
            self.compare_table.delete(item)

        runs = [self.runs[n] for n in ALGORITHMS[:3] if n in self.runs]
        for run in runs:
            s = run["summary"]
            self.compare_table.insert(
                "", "end",
                values=(run["label"], f"{s['avg_wt']:.2f}", f"{s['avg_tat']:.2f}",
                        f"{s['util']:.2f}", f"{s['throughput']:.3f}", s["switches"]),
            )

        if len(runs) < 2:
            self._set_text(self.compare_text, "Run at least two algorithms to see which is best.")
            return

        def best(metric, title):
            low = min(r["summary"][metric] for r in runs)
            winners = [r["label"] for r in runs if abs(r["summary"][metric] - low) < 1e-9]
            return f"{title}: {' / '.join(winners)}"

        self._set_text(
            self.compare_text,
            best("avg_wt", "Lowest average waiting time") + "\n"
            + best("avg_tat", "Lowest average turnaround time") + "\n"
            + best("switches", "Fewest context switches"),
        )


def main():
    root = tk.Tk()
    SchedulerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()