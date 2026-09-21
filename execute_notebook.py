# -*- coding: utf-8 -*-
"""
Script thực thi notebook.ipynb, chụp toàn bộ output stdout và biểu đồ matplotlib (PNG base64)
và lưu trực tiếp vào notebook.ipynb để notebook có đầy đủ kết quả hiển thị.
"""
import os
import sys
import io
import base64
import warnings
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress the expected "FigureCanvasAgg is non-interactive" warning that fires
# every time plt.show() is called under the Agg backend.  We capture figures
# manually via fig.savefig(), so this warning is purely noise.
# Use module='' (matches any module) so the filter also covers warnings raised
# from exec()'d notebook cell code (which shows up as module '<string>').
warnings.filterwarnings(
    'ignore',
    message='.*FigureCanvasAgg is non-interactive.*',
    category=UserWarning,
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def execute_notebook(nb_path):
    print(f"[*] Bắt đầu thực thi notebook: {nb_path}")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Make the warnings filter visible inside exec()'d cell code
    global_env = {
        '__name__': '__main__',
        '__builtins__': __builtins__,
    }
    exec_count = 1

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            src = cell.source
            first_line = src.strip().split('\n')[0] if src.strip() else ''
            print(f"[{exec_count:02d}] Đang chạy cell {idx:02d}: {first_line[:60]}...")
            
            # Đóng các figure cũ trước khi chạy cell mới
            plt.close('all')
            
            # Bắt stdout
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output
            
            cell.outputs = []
            try:
                exec(src, global_env)
            except Exception as e:
                sys.stdout = old_stdout
                print(f"[LỖI TẠI CELL {idx}]: {e}")
                import traceback
                traceback.print_exc()
                raise e
            finally:
                sys.stdout = old_stdout

            # Thu thập stdout
            out_text = redirected_output.getvalue()
            if out_text:
                cell.outputs.append(nbf.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=out_text
                ))

            # Thu thập biểu đồ matplotlib nếu có
            fignums = plt.get_fignums()
            for fignum in fignums:
                fig = plt.figure(fignum)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                b64_data = base64.b64encode(buf.read()).decode('ascii')
                buf.close()
                plt.close(fig)

                cell.outputs.append(nbf.v4.new_output(
                    output_type='display_data',
                    data={
                        'image/png': b64_data,
                        'text/plain': f'<Figure size {fig.get_size_inches()} with axes>'
                    },
                    metadata={}
                ))

            cell.execution_count = exec_count
            exec_count += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"[HOÀN TẤT] Đã thực thi thành công {exec_count-1} code cells và lưu đầy đủ outputs & biểu đồ vào {nb_path}!")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nb_file = os.path.join(script_dir, 'notebook.ipynb')
    execute_notebook(nb_file)
