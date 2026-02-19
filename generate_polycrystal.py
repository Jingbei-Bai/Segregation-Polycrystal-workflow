#!/usr/bin/env python3
"""
generate_polycrystal.py

generate_polycrystal(box, random_val, out='polycrystal.txt', ...)
"""

from pathlib import Path
from typing import Sequence, Union
import subprocess
import shutil
import re


def _normalize_box_input(box: Union[Sequence, int, float, str]):
    """接受单个标量（int/float/str）或序列（长度1或3），返回三个字符串表示的 box 大小。"""
    # 如果是单个标量（非序列或是字符串/数字），统一处理为三份相同的值
    if isinstance(box, (int, float)):
        s = str(box)
        return [s, s, s]
    if isinstance(box, str):
        # 允许传入像 "70" 或 "70 80 90" 的字符串
        parts = box.split()
        if len(parts) == 1:
            return [parts[0], parts[0], parts[0]]
        if len(parts) == 3:
            return parts
        raise ValueError('字符串形式的 box 必须是一个值或三个用空格分隔的值，例如 "70" 或 "70 80 90"')
    # 如果是序列（list/tuple）
    try:
        seq = list(box)
    except TypeError:
        raise ValueError('box 必须是数字/字符串或序列')

    if len(seq) == 1:
        val = seq[0]
        return [str(val), str(val), str(val)]
    if len(seq) == 3:
        return [str(seq[0]), str(seq[1]), str(seq[2])]
    raise ValueError('参数 box 必须是 1 个值或 3 个值，例如: 70 或 [70,70,70]')


def write_polycrystal(path: Path, box_vals, random_val: int):
    # 写入 polycrystal.txt（覆盖）
    lines = []
    lines.append('box {} {} {}'.format(box_vals[0], box_vals[1], box_vals[2]))
    lines.append('random {}'.format(random_val))
    content = "\n".join(lines) + "\n"
    # 确保父目录存在
    parent = path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f'无法创建父目录 {parent}: {e}')

    # 如果目标路径已经存在且是一个目录，给出明确提示（Windows 上尝试以文件打开目录会导致 Errno 22）
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f'目标路径是一个文件夹，无法写入文件: {path}。请换一个文件名或删除同名文件夹。')

    try:
        path.write_text(content, encoding='utf-8')
    except OSError as e:
        # 捕获并包装错误，提供上下文信息
        raise OSError(f'写入文件 {path} 失败: {e}')


def _sanitize_filename(name: str) -> str:
    """对文件名做简单清洗，移除 Windows/UNIX 上常见的非法字符并避免保留名。

    返回一个安全的文件名（不包含路径）。"""
    if not isinstance(name, str) or name == '':
        return 'polycrystal.txt'

    # 只保留 basename
    base = Path(name).name
    # 移除控制字符
    base = re.sub(r'[\x00-\x1f\x7f]', '', base)
    # 移除这些非法字符：<>:"/\\|?*
    base = re.sub(r'[<>:"/\\|?*]', '_', base)
    # 处理 Windows 保留名（CON, PRN, AUX, NUL, COM1..COM9, LPT1..LPT9）
    reserved = { 'CON','PRN','AUX','NUL' } | {f'COM{i}' for i in range(1,10)} | {f'LPT{i}' for i in range(1,10)}
    stem = Path(base).stem.upper()
    if stem in reserved:
        base = f'_{base}'
    # 如果清洗后为空，则使用默认名
    if base == '' or base in ('.', '..'):
        base = 'polycrystal.txt'
    return base


def generate_polycrystal(box: Union[Sequence, int, float, str], random_val: int, out: Union[str, Path] = 'polycrystal.txt', xsf_path: Union[str, Path] = 'aluminium.xsf', show_preview: bool = True, print_command: bool = True):
    """生成 polycrystal 文件的函数化接口。

    参数:
      box: 单个值、字符串或长度为1/3的序列；例如 70 或 '70 80 90' 或 [70] 或 [70,80,90]
      random_val: random 后面的整数
      out: 输出文件路径（默认 'polycrystal.txt'）
      xsf_path: 用于提示的 aluminium.xsf 路径（默认 'aluminium.xsf'）
      show_preview: 是否在完成后打印文件内容
      print_command: 是否打印建议的 atomsk 命令

    返回: 输出文件的 Path 对象
    """
    box_vals = _normalize_box_input(box)
    # 清洗并规范输出路径：仅清洗文件名部分，保留用户可能指定的目录
    out_p = Path(out)
    try:
        parent = out_p.parent
    except Exception:
        parent = Path('.')
    safe_name = _sanitize_filename(out_p.name)
    out_path = parent / safe_name
    if str(out_path) != str(out_p):
        # 提示用户我们对文件名做了修正
        if show_preview:
            print(f"注意：输出文件名已被清洗为：{out_path.name}")

    try:
        write_polycrystal(out_path, box_vals, int(random_val))
    except Exception as e:
        # 写入失败时尝试回退到当前工作目录下的备份文件名
        import time, tempfile
        fallback_name = f'polycrystal_{int(time.time())}.txt'
        fallback_path = Path('.') / fallback_name
        try:
            write_polycrystal(fallback_path, box_vals, int(random_val))
        except Exception as e2:
            # 如果回退也失败，抛出原始错误（包含回退错误信息）
            raise OSError(f'尝试写入 {out_path} 失败（原因：{e}），且回退写入到 {fallback_path} 也失败（原因：{e2}）。')
        else:
            if show_preview:
                print(f'注意：无法写入指定路径 {out_path}，已回退并写入：{fallback_path.resolve()}')
            out_path = fallback_path

    if show_preview:
        print('已生成：{}'.format(out_path.resolve()))
        print('\n生成内容预览:')
        print(out_path.read_text(encoding='utf-8'))

    xsf_p = Path(xsf_path)
    if not xsf_p.exists():
        print('提示：当前目录中未发现 aluminium.xsf（{}）。请确保在运行 atomsk 命令时该文件路径正确或放在同一目录。'.format(xsf_p.resolve()))

    if print_command:
        print('\n下一步：在安装有 atomsk 的终端中运行以下命令以生成多晶：')
        print('atomsk --polycrystal {} {} polycrystal.lmp'.format(xsf_p.name, out_path.name))

    return out_path


# 替换原有的 __main__ 提示为 GUI 主函数（使用 tkinter）
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog, scrolledtext
except Exception:
    tk = None


def _confirm_and_delete_all_except(directory: Path, keep_name: str, log_fn=None):
    """删除 directory 下的所有文件/文件夹，但保留名字为 keep_name 的项。

    返回删除的路径列表。
    log_fn 可为函数（用于写入日志），若为 None 则忽略。
    """
    deleted = []
    for child in directory.iterdir():
        if child.name == keep_name:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            deleted.append(child)
            if log_fn:
                log_fn(f'Deleted: {child}')
        except Exception as e:
            if log_fn:
                log_fn(f'Failed to delete {child}: {e}')
    return deleted


def main_gui():
    """启动一个简单的 Tkinter 界面，让用户输入 box、random 和文件路径，并生成 polycrystal 文件。

    新增功能：可以选择在生成后运行 atomsk，并在 atomsk 成功后（或直接）删除除 polycrystal.lmp 外的所有文件（需要用户二次确认）。
    如果系统没有 tkinter（tk 为 None），将打印基于控制台的提示。
    """
    if tk is None:
        print('当前 Python 环境中不可用 tkinter，无法启动 GUI。')
        print('请直接从 Python 导入并调用 generate_polycrystal(box, random_val, out, xsf_path)')
        return

    root = tk.Tk()
    root.title('Polycrystal 生成器')

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill=tk.BOTH, expand=True)

    tk.Label(frm, text='Box (单值或三个值，例如: 70 或 70 80 90)：').grid(row=0, column=0, sticky='w')
    ent_box = tk.Entry(frm, width=30)
    ent_box.insert(0, '70')
    ent_box.grid(row=0, column=1, sticky='w')

    tk.Label(frm, text='Random (整数)：').grid(row=1, column=0, sticky='w')
    ent_random = tk.Entry(frm, width=30)
    ent_random.insert(0, '20')
    ent_random.grid(row=1, column=1, sticky='w')

    tk.Label(frm, text='输出文件 (polycrystal.txt)：').grid(row=2, column=0, sticky='w')
    ent_out = tk.Entry(frm, width=30)
    ent_out.insert(0, 'polycrystal.txt')
    ent_out.grid(row=2, column=1, sticky='w')

    def browse_out():
        p = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text files','*.txt'),('All','*.*')], initialfile=ent_out.get())
        if p:
            ent_out.delete(0, tk.END)
            ent_out.insert(0, p)

    btn_browse_out = tk.Button(frm, text='浏览', command=browse_out)
    btn_browse_out.grid(row=2, column=2, padx=5)

    tk.Label(frm, text='aluminium.xsf 路径：').grid(row=3, column=0, sticky='w')
    ent_xsf = tk.Entry(frm, width=30)
    ent_xsf.insert(0, 'aluminium.xsf')
    ent_xsf.grid(row=3, column=1, sticky='w')

    def browse_xsf():
        p = filedialog.askopenfilename(filetypes=[('XSF files','*.xsf'),('All','*.*')])
        if p:
            ent_xsf.delete(0, tk.END)
            ent_xsf.insert(0, p)

    btn_browse_xsf = tk.Button(frm, text='浏览', command=browse_xsf)
    btn_browse_xsf.grid(row=3, column=2, padx=5)

    # 运行 atomsk 与清理选项
    run_atomsk_var = tk.BooleanVar(value=False)
    clean_var = tk.BooleanVar(value=False)
    chk_run = tk.Checkbutton(frm, text='生成后自动运行 atomsk', variable=run_atomsk_var)
    chk_run.grid(row=6, column=1, sticky='w')
    chk_clean = tk.Checkbutton(frm, text='生成后删除除 polycrystal.lmp 外的所有文件（需确认）', variable=clean_var)
    chk_clean.grid(row=7, column=1, sticky='w')

    # 日志/预览文本框
    tk.Label(frm, text='日志 / 生成内容预览：').grid(row=4, column=0, sticky='nw', pady=(10,0))
    txt_preview = scrolledtext.ScrolledText(frm, width=80, height=12)
    txt_preview.grid(row=4, column=1, columnspan=2, pady=(10,0))

    def log(msg: str):
        txt_preview.insert(tk.END, msg + '\n')
        txt_preview.see(tk.END)

    def on_generate():
        txt_preview.delete('1.0', tk.END)
        box_str = ent_box.get().strip()
        rnd_str = ent_random.get().strip()
        out_path = ent_out.get().strip()
        xsf_path = ent_xsf.get().strip()

        # 解析并校验
        try:
            box_vals = _normalize_box_input(box_str)
        except Exception as e:
            messagebox.showerror('参数错误', f'Box 参数解析失败：{e}')
            return

        try:
            rnd = int(rnd_str)
        except Exception:
            messagebox.showerror('参数错误', 'Random 必须是整数')
            return

        try:
            outp = generate_polycrystal(box_vals, rnd, out=out_path, xsf_path=xsf_path, show_preview=False, print_command=False)
        except Exception as e:
            messagebox.showerror('写入失败', f'生成文件时出错：{e}')
            return

        # 显示生成内容
        try:
            content = outp.read_text(encoding='utf-8')
            log('=== 生成内容 ===')
            log(content.strip())
        except Exception as e:
            log(f'无法读取生成文件：{e}')

        # 如果勾选运行 atomsk
        dir_for_ops = Path(outp).parent.resolve()
        atomsk_ran = False
        pre_existing = set(dir_for_ops.iterdir())
        created = set()
        if run_atomsk_var.get():
            cmd = ['atomsk', '--polycrystal', xsf_path, str(outp), 'polycrystal.lmp', '-wrap']
            log('运行 atomsk: ' + ' '.join(cmd))
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(dir_for_ops))
            except FileNotFoundError:
                messagebox.showerror('执行失败', '未找到 atomsk。请确保 atomsk 已安装并在 PATH 中。')
                log('Error: atomsk not found in PATH')
                return
            except Exception as e:
                messagebox.showerror('执行失败', f'运行 atomsk 时出错：{e}')
                log(f'运行 atomsk 异常：{e}')
                return

            log('--- atomsk stdout ---')
            log(proc.stdout.strip())
            log('--- atomsk stderr ---')
            log(proc.stderr.strip())

            if proc.returncode != 0:
                messagebox.showerror('atomsk 失败', f'atomsk 返回非零退出码：{proc.returncode}\n请查看日志以获取详细信息。')
                return
            atomsk_ran = True
            log('atomsk 执行成功，已生成 polycrystal.lmp（若 atomsk 支持此操作）')
            post_existing = set(dir_for_ops.iterdir())
            created = post_existing - pre_existing
            # 显示创建的文件列表
            if created:
                log('atomsk 本次运行创建了下面的文件/文件夹：')
                for p in sorted(created):
                    log(f'  {p.name}')
            else:
                log('atomsk 本次运行未检测到新创建的文件')

        # 如果勾选清理
        if clean_var.get():
            # 清理逻辑：
            # - 如果本次运行了 atomsk，删除 atomsk 本次创建的文件（除 polycrystal.lmp）以及本次使用的 polycrystal.txt
            # - 如果本次未运行 atomsk，只询问并只删除本次使用的 polycrystal.txt（不删除其他文件）
            if atomsk_ran:
                created_list = list(created)
                # 列出将要删除的目标（排除 .lmp）
                to_delete = [p for p in created_list if p.name != 'polycrystal.lmp']
                # 确保也删除本次使用的 polycrystal.txt
                if Path(outp).exists() and Path(outp) not in to_delete:
                    to_delete.append(Path(outp))

                if not to_delete:
                    messagebox.showinfo('清理', '没有发现需要删除的 atomsk 生成文件，只有 polycrystal.txt 将被删除（若存在）。')
                    # 仅删除 polycrystal.txt（如果存在）
                    if Path(outp).exists():
                        Path(outp).unlink()
                        log(f'Deleted: {outp}')
                        messagebox.showinfo('清理完成', f'已删除: {outp}')
                    else:
                        messagebox.showinfo('清理', '未检测到 polycrystal.txt，未删除任何文件')
                else:
                    preview_list = '\n'.join([p.name for p in to_delete[:50]])
                    if len(to_delete) > 50:
                        preview_list += '\n...(还有 %d 个项目)...' % (len(to_delete) - 50)

                    confirm = messagebox.askyesno('确认删除', f'将删除 atomsk 本次运行创建的以下文件（保留 polycrystal.lmp）：\n\n{preview_list}\n\n是否继续？ 此操作不可撤销。')
                    if confirm:
                        log('开始删除 atomsk 生成的文件...')
                        deleted = []
                        for p in to_delete:
                            try:
                                if p.is_dir():
                                    shutil.rmtree(p)
                                else:
                                    p.unlink()
                                deleted.append(p)
                                log(f'Deleted: {p.name}')
                            except Exception as e:
                                log(f'Failed to delete {p}: {e}')
                        messagebox.showinfo('清理完成', f'已删除 {len(deleted)} 项（排除 polycrystal.lmp）')
                    else:
                        log('用户取消了 atomsk 生成文件的删除')
            else:
                # 未运行 atomsk，本次仅删除 polycrystal.txt（需用户确认）
                if Path(outp).exists():
                    confirm = messagebox.askyesno('确认删除', f'未检测到 atomsk 运行。是否只删除本次生成的 polycrystal 文件：\n{outp} ?')
                    if confirm:
                        try:
                            Path(outp).unlink()
                            log(f'Deleted: {outp}')
                            messagebox.showinfo('清理完成', f'已删除: {outp}')
                        except Exception as e:
                            messagebox.showerror('删除失败', f'删除 {outp} 时失败：{e}')
                    else:
                        log('用户取消了仅删除 polycrystal.txt 的操作')
                else:
                    messagebox.showinfo('清理', 'polycrystal.txt 不存在，未删除任何文件')

        # After cleanup (if any), convert polycrystal.lmp to a dump using atomsk and save both files
        try:
            poly_lmp_path = dir_for_ops / 'polycrystal.lmp'
            if poly_lmp_path.exists():
                cmd_dump = ['atomsk', poly_lmp_path.name, 'dump']
                log('运行 atomsk 将 polycrystal.lmp 转换为 dump: ' + ' '.join(cmd_dump))
                try:
                    proc_dump = subprocess.run(cmd_dump, capture_output=True, text=True, cwd=str(dir_for_ops))
                except FileNotFoundError:
                    log('Error: 未找到 atomsk，可选地请安装 atomsk 并将其加入 PATH，跳过 dump 生成。')
                except Exception as e:
                    log(f'运行 atomsk 转换为 dump 时发生异常: {e}')
                else:
                    log('--- atomsk(dump) stdout ---')
                    log(proc_dump.stdout.strip())
                    log('--- atomsk(dump) stderr ---')
                    log(proc_dump.stderr.strip())
                    if proc_dump.returncode != 0:
                        log(f'atomsk 转换返回非零退出码: {proc_dump.returncode} (请查看 stderr)')
                    else:
                        log('atomsk 转换为 dump 成功，polycrystal.lmp 与生成的 dump 已保存在：' + str(dir_for_ops))
            else:
                log('polycrystal.lmp 未找到，跳过转换为 dump 步骤')
        except Exception as e:
            log(f'尝试生成 dump 时发生未预料的错误: {e}')

    btn_generate = tk.Button(frm, text='生成 polycrystal.txt', command=on_generate, bg='#4caf50', fg='white')
    btn_generate.grid(row=5, column=1, pady=10, sticky='w')

    btn_quit = tk.Button(frm, text='退出', command=root.destroy)
    btn_quit.grid(row=5, column=2, pady=10, sticky='e')

    root.mainloop()


if __name__ == '__main__':
    main_gui()
