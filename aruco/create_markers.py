import tempfile
from pathlib import Path
import cv2
from fpdf import FPDF
from PIL import Image
import utils

def create_markers(marker_type):
    # Get marker IDs
    marker_ids = utils.get_marker_ids(marker_type)
    if marker_type == 'robots':
        marker_ids = 5 * marker_ids + marker_ids[:4]
    elif marker_type == 'cubes':
        marker_ids = [marker_id for marker_id in marker_ids[:8] for _ in range(6)]
    elif marker_type == 'corners':
        marker_ids = 3 * marker_ids + marker_ids[:15]

    # Paper params
    output_dir = 'printouts'
    pdf_name = 'markers-{}.pdf'.format(marker_type)
    orientation = 'P'
    paper_params = utils.get_paper_params(orientation)

    # Marker params
    marker_length = 0.018  # 18 mm
    marker_length_pixels = 6
    sticker_length_mm = {'robots': 25, 'cubes': 28, 'corners': 24}[marker_type]
    sticker_spacing_mm = 3
    marker_length_mm = 1000 * marker_length
    scale_factor = marker_length_mm / (paper_params['mm_per_printed_pixel'] * marker_length_pixels)
    stickers_per_row = int((paper_params['width_mm'] - 2 * paper_params['margin_mm'] + sticker_spacing_mm) / (sticker_length_mm + sticker_spacing_mm))
    aruco_dict = cv2.aruco.Dictionary_get(utils.get_marker_dict_id())

    # Create PDF
    pdf = FPDF(orientation, 'mm', 'letter')
    pdf.add_page()
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        for i, marker_id in enumerate(marker_ids):
            image_path = str(Path(tmp_dir_name) / '{}.png'.format(marker_id))
            Image.fromarray(cv2.aruco.drawMarker(aruco_dict, marker_id, int(scale_factor * marker_length_pixels))).save(image_path)
            center_x = paper_params['margin_mm'] + sticker_length_mm / 2 + (sticker_length_mm + sticker_spacing_mm) * (i % stickers_per_row)
            center_y = paper_params['margin_mm'] + sticker_length_mm / 2 + (sticker_length_mm + sticker_spacing_mm) * (i // stickers_per_row)
            pdf.rect(
                x=(center_x - sticker_length_mm / 2 - pdf.line_width / 2),
                y=(center_y - sticker_length_mm / 2 - pdf.line_width / 2),
                w=(sticker_length_mm + pdf.line_width),
                h=(sticker_length_mm + pdf.line_width)
            )
            pdf.image(image_path, x=(center_x - marker_length_mm / 2), y=(center_y - marker_length_mm / 2), w=marker_length_mm, h=marker_length_mm)

    # Save PDF
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    pdf.output(output_dir / pdf_name)

if __name__ == '__main__':
    create_markers('robots')
    create_markers('cubes')
    create_markers('corners')
