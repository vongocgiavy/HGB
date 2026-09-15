"""
Wrapper cho build_notebook.py nhằm duy trì tính tương thích ngược (Backward Compatibility).
Khuyến nghị sử dụng trực tiếp: python build_notebook.py
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("build_notebook", run_name="__main__")
