import tempfile
from pathlib import Path
import cv2
from fpdf import FPDF
from PIL import Image
import utils

# Paper params
output_dir = 'printouts'
pdf_name = 'charuco-board.pdf'
orientation = 'L'
paper_params = utils.get_paper_params(orientation)

# Board params
board_params = utils.get_charuco_board_params()
square_length_pixels = 24
board_length_pixels = board_params['squares_x'] * square_length_pixels
board_width_pixels = board_params['squares_y'] * square_length_pixels
board_length_mm = 1000 * board_params['squares_x'] * board_params['square_length']
board_width_mm = 1000 * board_params['squares_y'] * board_params['square_length']
scale_factor = board_length_mm / (paper_params['mm_per_printed_pixel'] * board_length_pixels)
aruco_dict = cv2.aruco.Dictionary_get(utils.get_marker_dict_id())

# Create board image
board = cv2.aruco.CharucoBoard_create(board_params['squares_x'], board_params['squares_y'], board_params['square_length'], board_params['marker_length'], aruco_dict)
board_image = board.draw((int(scale_factor * board_length_pixels), int(scale_factor * board_width_pixels)))

# Create PDF
pdf = FPDF(orientation, 'mm', 'letter')
pdf.add_page()
with tempfile.TemporaryDirectory() as tmp_dir_name:
    image_path = str(Path(tmp_dir_name) / 'board.png')
    Image.fromarray(board_image).save(image_path)
    pdf.image(image_path, x=(paper_params['width_mm'] - board_length_mm) / 2, y=(paper_params['height_mm'] - board_width_mm) / 2, w=board_length_mm, h=board_width_mm)

# Save PDF
output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir(parents=True)
pdf.output(output_dir / pdf_name)
