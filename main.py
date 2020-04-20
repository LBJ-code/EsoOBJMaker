import argparse
from GUI import GUIController
from DicomProcessor import DicomProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--dicom_dir", type=str, required=True, help="DICOMファイルへのパス")
parser.add_argument("--output_dir", type=str, default="./tmp", help='OBJファイルの出力先を指定')
parser.add_argument("--resized_button_height", type=int, default=154, help="メニューボタンの高さ")
parser.add_argument("--resized_button_width", type=int, default=314, help="メニューボタンの幅")
parser.add_argument("--eso_radius", type=float, default=15.0, help='mm単位で食道モデルの半径を指定')
parser.add_argument("--circle_divisions", type=int, default=50, help='食道モデルの半径方向への分解能')
parser.add_argument("--spline_num", type=int, default=10, help='食道の中心点間を補完する点の数')
parser.add_argument("--magnification_ratio", type=int, default=4, help='ダブルクリック時の拡大倍率（偶数のほうがいいかも）')
args = parser.parse_args()

if __name__ == '__main__':
    GUI = GUIController(args)
    GUI.run()