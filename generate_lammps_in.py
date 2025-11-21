#!/usr/bin/env python3
"""
generate_lammps_in.py

生成 LAMMPS 输入文件（in 文件）。提供函数接口 `generate_lammps_in(...)` 和一个简单的 Tkinter GUI（可视化窗口）用于交互式输入。

用户需要指定三个必输项：
1) read_data 后面的 lmp 文件名（及可选附加参数，比如 `extra/atom/types 1`）
2) print append 后面的 txt 文件名
3) pair_coeff 行中使用的 eam 文件（例如 AlCu.eam）和可选物种名

脚本不会执行 LAMMPS，只会生成文本文件并在 GUI 中显示建议运行命令。
"""

from pathlib import Path
from typing import Sequence, Union
import subprocess
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
except Exception:
    tk = None

# add: import sorter utility
try:
    from dump_utils import sort_dump_file
except Exception:
    sort_dump_file = None


DEFAULT_TEMPLATE = '''units           metal
atom_style      atomic
boundary        p p p
read_data       {read_data}
mass 1 {mass1}
mass 2 {mass2}
pair_style  eam/alloy
pair_coeff * * {eam_file} {species_line}
variable total_atoms equal count(all)
timestep {timestep}
thermo {thermo}
thermo_style        {thermo_style}
minimize 0 1e-8 10000 10000
write_dump all atom dump.atom
variable atom_id loop ${{total_atoms}}
label loop_start

set atom ${{atom_id}} type 2

run 0

variable totalenergy equal "pe"

print "${{atom_id}} ${{totalenergy}}" append {append_txt}
set atom ${{atom_id}} type 1

next atom_id
jump SELF loop_start
run 0
print "Done!"
#mpiexec -np 4 lmp -in in -pk omp 4 -sf omp
'''


def generate_lammps_in(read_data: str, append_txt: str, eam_file: str, out: Union[str, Path] = 'in.lmp', species: Union[Sequence[str], str] = ('Al', 'Cu'), mass1: Union[str,float] = '26.98', mass2: Union[str,float] = '63.55', timestep: float = 0.001, thermo: int = 1000, thermo_style: str = 'custom step pe', read_data_extra: str = '') -> Path:
    """生成 LAMMPS in 文件并返回输出路径。

    参数:
      read_data: read_data 后面的参数（可以包含路径和额外参数），例如 'medium_final_atoms.lmp extra/atom/types 1'
      append_txt: print append 后的文件名，例如 'medium_Al_Cu_eam4.txt'
      eam_file: pair_coeff 行中的 eam 文件名，例如 'AlCu.eam'
      out: 输出 in 文件路径（默认 'in.lmp')
      species: 物种名序列或用空格分隔的字符串（默认 ('Al','Cu')）
      mass1/mass2: 元素质量（默认示例中的值）
      timestep, thermo, thermo_style: LAMMPS 参数
      read_data_extra: 可选的额外 read_data 参数（如果不包含在 read_data 参数中）

    返回: 写入的 Path 对象
    """
    # 构建 species 行
    if isinstance(species, (list, tuple)):
        species_line = ' '.join(map(str, species))
    else:
        species_line = str(species)

    # 合并 read_data 字段（如果 read_data_extra 提供，则合并）
    read_data_full = (read_data + ' ' + read_data_extra).strip()

    content = DEFAULT_TEMPLATE.format(
        read_data=read_data_full,
        mass1=str(mass1),
        mass2=str(mass2),
        eam_file=str(eam_file),
        species_line=species_line,
        timestep=timestep,
        thermo=thermo,
        thermo_style=thermo_style,
        append_txt=str(append_txt),
    )

    out_path = Path(out)
    out_path.write_text(content, encoding='utf-8')
    return out_path


def main_gui():
    """简单的 Tkinter GUI，供交互式生成 in 文件。"""
    if tk is None:
        print('当前 Python 环境中不可用 tkinter，无法启动 GUI。')
        print('请直接从 Python 导入并调用 generate_lammps_in(...)')
        return

    root = tk.Tk()
    root.title('LAMMPS in 生成器')

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill=tk.BOTH, expand=True)

    tk.Label(frm, text='read_data (例如: medium_final_atoms.lmp extra/atom/types 1):').grid(row=0, column=0, sticky='w')
    ent_read = tk.Entry(frm, width=50)
    ent_read.insert(0, 'medium_final_atoms.lmp extra/atom/types 1')
    ent_read.grid(row=0, column=1, sticky='w')

    tk.Label(frm, text='append 文件名 (例如: medium_Al_Cu_eam4.txt):').grid(row=1, column=0, sticky='w')
    ent_append = tk.Entry(frm, width=50)
    ent_append.insert(0, 'medium_Al_Cu_eam4.txt')
    ent_append.grid(row=1, column=1, sticky='w')

    tk.Label(frm, text='EAM 文件 (例如: AlCu.eam):').grid(row=2, column=0, sticky='w')
    ent_eam = tk.Entry(frm, width=50)
    ent_eam.insert(0, 'AlCu.eam')
    ent_eam.grid(row=2, column=1, sticky='w')

    tk.Label(frm, text='物种（空格分隔，默认: Al Cu）:').grid(row=3, column=0, sticky='w')
    ent_species = tk.Entry(frm, width=50)
    ent_species.insert(0, 'Al Cu')
    ent_species.grid(row=3, column=1, sticky='w')

    tk.Label(frm, text='输出 in 文件 (例如: in.lmp):').grid(row=4, column=0, sticky='w')
    ent_out = tk.Entry(frm, width=50)
    ent_out.insert(0, 'in.lmp')
    ent_out.grid(row=4, column=1, sticky='w')

    tk.Label(frm, text='可选: read_data 额外参数 (可留空):').grid(row=5, column=0, sticky='w')
    ent_read_extra = tk.Entry(frm, width=50)
    ent_read_extra.grid(row=5, column=1, sticky='w')

    # 新增选项：是否生成后自动运行 LAMMPS，以及是否让 in 名称与 append 文件名自动一致
    run_lammps_var = tk.BooleanVar(value=False)
    sync_names_var = tk.BooleanVar(value=False)
    chk_run_lammps = tk.Checkbutton(frm, text='生成后自动运行 LAMMPS (mpiexec -np 4 lmp -in <in> -pk omp 4 -sf omp)', variable=run_lammps_var)
    chk_run_lammps.grid(row=5, column=2, sticky='w', padx=(10,0))
    chk_sync = tk.Checkbutton(frm, text='使 -in 后的文件名与 append 输出名自动一致 (append = in 的 basename + .txt)', variable=sync_names_var)
    chk_sync.grid(row=6, column=2, sticky='w', padx=(10,0))

    # 预览/日志
    tk.Label(frm, text='生成内容预览：').grid(row=6, column=0, sticky='nw', pady=(10,0))
    txt_preview = scrolledtext.ScrolledText(frm, width=80, height=20)
    txt_preview.grid(row=6, column=1, columnspan=2, pady=(10,0))

    def log(msg: str):
        txt_preview.insert(tk.END, msg + '\n')
        txt_preview.see(tk.END)

    def browse_read():
        p = filedialog.askopenfilename(filetypes=[('LMP files','*.lmp'),('All','*.*')])
        if p:
            ent_read.delete(0, tk.END)
            ent_read.insert(0, p)

    def browse_eam():
        p = filedialog.askopenfilename(filetypes=[('EAM files','*.eam'),('All','*.*')])
        if p:
            ent_eam.delete(0, tk.END)
            ent_eam.insert(0, p)

    btn_browse_read = tk.Button(frm, text='浏览 read_data', command=browse_read)
    btn_browse_read.grid(row=0, column=2, padx=5)
    btn_browse_eam = tk.Button(frm, text='浏览 eam', command=browse_eam)
    btn_browse_eam.grid(row=2, column=2, padx=5)

    def on_generate():
        txt_preview.delete('1.0', tk.END)
        rd = ent_read.get().strip()
        ap = ent_append.get().strip()
        eam = ent_eam.get().strip()
        sp = ent_species.get().strip()
        out = ent_out.get().strip()
        rd_extra = ent_read_extra.get().strip()

        if not rd:
            messagebox.showerror('参数错误', 'read_data 不能为空')
            return
        if not ap and not sync_names_var.get():
            messagebox.showerror('参数错误', 'append 文件名不能为空（或启用自动一致选项）')
            return
        if not eam:
            messagebox.showerror('参数错误', 'eam 文件不能为空')
            return

        species_list = sp.split()

        # 如果勾选了同步名字，则把 append 文件名设置为 in 的 basename + .txt
        append_to_use = ap
        if sync_names_var.get():
            try:
                append_to_use = str(Path(out).stem) + '.txt'
                # 更新界面显示
                ent_append.delete(0, tk.END)
                ent_append.insert(0, append_to_use)
            except Exception:
                append_to_use = ap

        try:
            # 先生成 in 文件（如果同步名，generate_lammps_in 会使用 append_to_use）
            outp = generate_lammps_in(rd, append_to_use, eam, out=out, species=species_list, read_data_extra=rd_extra)
        except Exception as e:
            messagebox.showerror('写入失败', f'生成 in 文件时出错：{e}')
            return

        txt_preview.insert(tk.END, outp.read_text(encoding='utf-8'))

        # 如果勾选运行 LAMMPS
        if run_lammps_var.get():
            dir_for_ops = Path(outp).parent.resolve()
            in_basename = Path(outp).name
            cmd = ['mpiexec', '-np', '4', 'lmp', '-in', in_basename, '-pk', 'omp', '4', '-sf', 'omp']
            # run in background thread to avoid blocking GUI
            def _run_lammps():
                def safe_log(s: str):
                    # use after with function+args to avoid lambda-related linter warnings
                    root.after(0, log, s)

                safe_log('\nRunning LAMMPS with: ' + ' '.join(cmd))
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(dir_for_ops))
                except FileNotFoundError:
                    root.after(0, messagebox.showerror, '执行失败', '未找到 mpiexec 或 lmp 可执行文件。请确保 mpiexec 和 lmp 在 PATH 中。')
                    safe_log('Error: mpiexec or lmp not found in PATH')
                    return
                except Exception as e:
                    root.after(0, messagebox.showerror, '执行失败', f'运行 LAMMPS 时出错：{e}')
                    safe_log(f'运行 LAMMPS 异常：{e}')
                    return

                # Do not stream full LAMMPS stdout/stderr into the GUI preview to avoid
                # flooding the visualization pane. Only report high-level status.
                safe_log('LAMMPS finished (see log.lammps in the working directory for full output if needed)')

                if proc.returncode != 0:
                    root.after(0, messagebox.showerror, 'LAMMPS 失败', f'LAMMPS 返回非零退出码：{proc.returncode}\n请查看日志以获取详细信息。')
                    return

                # 尝试删除 LAMMPS 默认生成的日志文件 log.lammps
                safe_log('LAMMPS 运行成功，正在尝试删除 log.lammps（如果存在）...')
                try:
                    log_path = dir_for_ops / 'log.lammps'
                    if log_path.exists():
                        safe_log(f'发现日志文件：{log_path.name}，尝试删除...')
                        log_path.unlink()
                        safe_log(f'已删除日志文件：{log_path.name}')
                    else:
                        safe_log('未发现 log.lammps，无需删除')
                except Exception as e:
                    safe_log(f'删除 log.lammps 失败：{e}')
                    root.after(0, messagebox.showwarning, '清理日志失败', f'运行完成但无法删除 log.lammps：{e}')

                # After successful run, attempt to sort dump.atom (if present).
                try:
                    dump_path = dir_for_ops / 'dump.atom'
                    if dump_path.exists() and sort_dump_file is not None:
                        # call sorter without a GUI logger to avoid streaming logs into preview
                        sort_dump_file(dump_path, header_lines=9, logger=None)
                        root.after(0, messagebox.showinfo, 'LAMMPS 完成', 'LAMMPS 运行完成；若存在 dump.atom，已自动排序。')
                    else:
                        root.after(0, messagebox.showinfo, 'LAMMPS 完成', 'LAMMPS 运行完成。')
                except Exception as e:
                    root.after(0, messagebox.showwarning, '排序失败', f'dump.atom 自动排序失败：{e}')

            threading.Thread(target=_run_lammps, daemon=True).start()

        messagebox.showinfo('已生成', f'已生成: {outp}\n请在含有 LAMMPS 的终端中使用该 in 文件运行 LAMMPS（如果未自动运行的话）')

    btn_generate = tk.Button(frm, text='生成 in 文件', command=on_generate, bg='#2196f3', fg='white')
    btn_generate.grid(row=7, column=1, pady=10, sticky='w')

    btn_quit = tk.Button(frm, text='退出', command=root.destroy)
    btn_quit.grid(row=7, column=2, pady=10, sticky='e')

    root.mainloop()


if __name__ == '__main__':
    main_gui()
