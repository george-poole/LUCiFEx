import os
import json
import glob
import natsort
from math import ceil


def heading(
    text: str, 
    depth: int,
    ):

    return f"{'#' * depth} {text}\n"

def table(
    table_rows: list[str],
) -> str:
    if not table_rows:
        return ''
    br = "\n"
    return f"""<table>
    {br.join(i for i in table_rows)}
    </table>{br}"""


def table_row(
    table_datas: list[str],
) -> str:  
    if not table_datas:
        return ''
    br = "\n"
    return f"""<tr>
    {br.join(i for i in table_datas)}
    </tr>"""


def table_data(
    caption_path: str,
    fig_dir_path: str,
    fig_name: str,
    caption: str,
    fig_ext: str = 'png',
    img_width: int = 400,
) -> str:
    return f"""<td>
    <img src="{fig_dir_path}/{fig_name}.{fig_ext}" width="{img_width}"/>
    <a href="{caption_path}">{caption}</a> 
    </td>"""


def extract_heading(
    file_path: str,
    default: str | None = None,
) -> str | None:
    
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook: dict[str, list[dict]] = json.load(f)

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            for line in cell.get('source', []):
                stripped = line.strip()
                if stripped.startswith('# '):
                    return stripped[2:].strip()
                
    return default


def make_gallery(
    gallery_file: str,
    html: bool,
    topheading: str,
    notebooks_dir_path: str,
    notebook_parts: list[tuple[str, str]],
    figures_dir_name: str,
    figures_ext: str,
    exclude: str | None = None
):
    markdown_string = heading(topheading, 1)

    for subheading, dir_name in notebook_parts:
        markdown_string = '\n'.join((markdown_string, heading(subheading, 2)))
        notebook_paths = natsort.natsorted(
            glob.glob(f'{notebooks_dir_path}/{dir_name}/*.ipynb')
        )
        if exclude is not None:
            notebook_paths = [nb for nb in notebook_paths if not exclude in nb]
        tds = []
        for ipynb_path in notebook_paths:
            fig_dir_path = os.path.join(os.path.dirname(ipynb_path), figures_dir_name)
            ipynb_name = os.path.basename(os.path.splitext(ipynb_path)[0])
            ipynb_heading = extract_heading(ipynb_path, 'ExtractHeadingError')
            if html: 
                caption_path = ipynb_path.replace('ipynb', 'html')
            else:
                caption_path = ipynb_path
            td = table_data(caption_path, fig_dir_path, ipynb_name, ipynb_heading, figures_ext)
            tds.append(td)
        Ntr = 3
        tds_divided = [tds[i * Ntr: Ntr * (i + 1)] for i in range(ceil(len(tds) / Ntr))]
        trs = [table_row(tdsd) for tdsd in tds_divided if tdsd]
        markdown_string = '\n'.join((markdown_string, table(trs)))
        
    with open(gallery_file, "w") as f:
        f.write(markdown_string)


if __name__ == "__main__":
    HEADING = 'Gallery'
    NOTEBOOKS_DIR_PATH = "./demo"
    THUMBNAIL_DIR_NAME = "thumbnails"
    THUMBNAIL_EXT = 'png'
    EXCLUDE = 'xxx'
    NOTEBOOK_PARTS = (
        ('Further Applications', 'P6_applications'),
        ('Convection', 'P5_convection'),
        ('Flow', 'P4_flow'),
        ('Transport', 'P3_transport'),
        ('Introductory', 'P2_introductory'),
    )
    OUTPUTS = (
        ('gallery.md', True), 
        ('gallery_local.md', False),
    )
    for file, html in OUTPUTS:
        make_gallery(
            file, 
            html,
            HEADING, 
            NOTEBOOKS_DIR_PATH, 
            NOTEBOOK_PARTS, 
            THUMBNAIL_DIR_NAME, 
            THUMBNAIL_EXT,
            EXCLUDE,
        )