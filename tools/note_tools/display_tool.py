# pip install PyMuPDF
import os
import fitz  # PyMuPDF
from tempfile import NamedTemporaryFile
import uuid
from IPython.display import Image, display

def tikz_to_image(
        tikz_code, 
        border=10,
        file_name=None, output_dir="assets", 
        dpi=300, 
        toSave=False, toLog=False, full_snippet=False, replace_sharp=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not file_name:
        file_name = str(uuid.uuid4()) + ".png"
    output_path = os.path.join(output_dir, file_name)

    # LaTeX模板，将TikZ代码嵌入其中
    latex_template = r"""
    \documentclass[tikz,border={border}pt]{{standalone}}
    \usepackage{{tikz}}
    \begin{{document}}
    # \centering
    {tikz_code}
    \end{{document}}
    """
    # comments
    if replace_sharp:
        tikz_code = tikz_code.replace("#", "%")

    # 将TikZ代码插入模板中
    if not full_snippet:
        latex_content = latex_template.format(
            tikz_code=tikz_code, 
            border=border,
        )
    else:
        latex_content = tikz_code
    if toLog:
        print(latex_content)

    # 创建一个临时目录和LaTeX文件
    with NamedTemporaryFile(suffix=".tex", delete=False) as tex_file:
        tex_path = tex_file.name
        tex_file.write(latex_content.encode("utf-8"))

    # 运行pdflatex将.tex文件转换为.pdf
    pdf_path = tex_path.replace(".tex", ".pdf")
    interaction = "nonstopmode" if toLog else "batchmode"
    # batchmode | nonstopmode | scrollmode | errorstopmode
    # --no-shell-escape 来关闭write18
    # os.system(f"pdflatex --no-shell-escape -interaction={interaction} -output-directory={os.path.dirname(tex_path)} {tex_path} > /dev/null 2>&1") # ' > /dev/null 2>&1' to suppress output
    os.system(f"pdflatex --no-shell-escape -interaction={interaction} -output-directory={os.path.dirname(tex_path)} {tex_path}")

    # 确保PDF生成成功
    if not os.path.exists(pdf_path):
        raise Exception("PDF生成失败。请检查TikZ代码或LaTeX环境。")

    # 使用PyMuPDF将PDF转换为图片
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(0)  # 加载第一页
    pix = page.get_pixmap(dpi=dpi)    # 设置分辨率
    img_data = pix.tobytes("png")      # 转换为PNG格式的二进制数据
    if toSave:
        pix.save(output_path)             # 保存为PNG文件
        print(f"图片已保存到 {output_path}")


    # 清理临时文件
    pdf_document.close()
    os.remove(tex_path)
    os.remove(pdf_path)

    display(Image(data=img_data))
    return img_data, output_path
    

    
from collections.abc import Iterable
def is_iterable(a):
    return isinstance(a, Iterable)
def nicely_print_matrix(matrix, row_labels=None, column_labels=None, name=None):
    EMPTY_STRING = ''
    if matrix is None or len(matrix) == 0:
        print('matrix is empty')
        return
    if name != None:
        print(name)
    if row_labels == None:
        row_labels = list(range(len(matrix)))
    if column_labels == None:
        if is_iterable(matrix[0]):
            column_labels = list(range(len(matrix[0])))
        else:
            column_labels = [EMPTY_STRING]
        # column_labels = list(range(len(matrix[0])))
    # print(row_labels)
    # print(column_labels)
    width = max([len(str(cell)) for row in matrix for cell in row])
    max_row_label_width = max([len(str(label)) for label in row_labels])
    max_column_label_width = max([len(str(label)) for label in column_labels])
    width = max(width, max_row_label_width, max_column_label_width)
    # x labels
    print(' ' * width, end=' ')
    for i in range(len(matrix[0])):
        print(str(column_labels[i]).rjust(width), end=' ')
    print()
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if j == 0:
                print(str(row_labels[i]).rjust(width), end=' ')
            print(str(matrix[i][j]).rjust(width), end=' ')
        print()