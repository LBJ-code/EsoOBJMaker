import numpy as np
import cv2
from DicomProcessor import DicomProcessor
from EsoOBJMaker import EsoOBJMaker

GUI_LOAD_NEXT = 1
GUI_LOAD_PREV = 0

class GUIController:

    def __init__(self, args):
        self.args = args
        self.Is_continue = True
        self.Is_zoomed = False
        self.dicom_processor = DicomProcessor(args.dicom_dir, args)
        self.dicom_size = len(self.dicom_processor.dicom_file_list)
        self.CT_img_for_show = self.dicom_processor.CT_for_imshow
        self.cur_src_CT_img = self.CT_img_for_show
        self.src_CT_size = self.cur_src_CT_img.shape
        self.zoomed_CT_size = self.preprocess_zoomed_CT_size()
        self.zoom_point = [0, 0]
        self.button_height, self.button_width = 0, 0
        self.button_img = self.create_button_img()
        # まだ画像のピクセル幅がx, yで同じときのみに対応している．
        self.img_eso_radius = args.eso_radius / self.dicom_processor.pixel_spacing_dx
        self.set_init_window()

        #出力用変数
        self.dicom_index = 0
        self.ijk_eso_centers = [None] * self.dicom_size

    def set_init_window(self):
        cv2.namedWindow('menu')
        cv2.namedWindow('CT image')

    def preprocess_zoomed_CT_size(self):
        zoomed_CT_size = [int(x / self.args.magnification_ratio) for x in self.CT_img_for_show.shape][:2]
        if zoomed_CT_size[0] % 2 is 0:
            zoomed_CT_size[0] -= 1
        if zoomed_CT_size[1] % 2 is 0:
            zoomed_CT_size[1] -= 1
        return zoomed_CT_size

    def create_button_img(self):
        buttons = ['back', 'next', 'export']
        button_size = (self.args.resized_button_width, self.args.resized_button_height)
        for i, button_name in enumerate(buttons):
            if i == 0:
                button_img = cv2.imread('./button_img/' + button_name + '.png')
                button_img = cv2.resize(button_img, button_size)
                self.button_height, self.button_width, _ = button_img.shape
            else:
                tmp = cv2.imread('./button_img/' + button_name + '.png')
                tmp = cv2.resize(tmp, button_size)
                button_img = cv2.hconcat([button_img, tmp])
        return button_img

    def change_cur_CT(self, mode):
        if mode is GUI_LOAD_NEXT:
            self.dicom_index += 1
        if mode is GUI_LOAD_PREV:
            self.dicom_index -= 1
        print('index : {}'.format(self.dicom_index))
        self.CT_img_for_show = self.dicom_processor.get_CT_by_index(self.dicom_index)
        self.cur_src_CT_img = self.CT_img_for_show
        self.Is_zoomed = False

    def GUI_imshow(self):
        cv2.imshow("menu", self.button_img)
        cv2.imshow("CT image", self.CT_img_for_show)
        cv2.waitKey(1)

    def menu_callbacks(self, event, x, y, flags, param):
        # click back
        if event == cv2.EVENT_LBUTTONUP and (0 <= x and x <= self.button_width) and self.dicom_index != 0:
            self.change_cur_CT(GUI_LOAD_PREV)

        # click next
        if event == cv2.EVENT_LBUTTONUP and (self.button_width < x and x < (2 * self.button_width)) and self.dicom_index != (self.dicom_size - 1):
            self.change_cur_CT(GUI_LOAD_NEXT)

        # click export
        if event == cv2.EVENT_LBUTTONUP and ((2 * self.button_width) < x and x < (3 * self.button_width)):
            print("click export")
            self.dicom_processor.make_distorted_dicom(self.ijk_eso_centers, self.img_eso_radius)
            eso_obj_maker = EsoOBJMaker(self.ijk_eso_centers, self.img_eso_radius,
                                        self.dicom_processor.calc_ijk2LPS_mat(),  self.args)
            eso_obj_maker.create_obj()
            self.Is_continue = False
            print('exit')

    def CT_image_callbacks(self, event, x, y, flags, param):
        if event is cv2.EVENT_LBUTTONUP and self.Is_zoomed is True:
            zoom_i_ratio, zoom_j_ratio = ((self.src_CT_size[0] / self.zoomed_CT_size[0]),
                                          (self.src_CT_size[1] / self.zoomed_CT_size[1]))
            cv2.circle(self.CT_img_for_show, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(self.CT_img_for_show, (x, y), int(self.img_eso_radius * zoom_i_ratio),
                       (200, 50, 50),lineType=cv2.LINE_AA)
            non_zoomed_i, non_zoomed_j = self.calc_non_zoomed_ij(x, y, zoom_i_ratio, zoom_j_ratio)
            print('clicked i:{} j:{} k:{}'.format(non_zoomed_i, non_zoomed_j,
                                                  self.dicom_processor.calc_k_on_ijk_coordinates(self.dicom_index)))
            self.ijk_eso_centers[self.dicom_index] = np.array([non_zoomed_i, non_zoomed_j,
                                                      self.dicom_processor.calc_k_on_ijk_coordinates(self.dicom_index)])
        elif event is cv2.EVENT_LBUTTONUP and self.Is_zoomed is False:
            cv2.circle(self.CT_img_for_show, (x, y), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(self.CT_img_for_show, (x, y), int(self.img_eso_radius), (200, 50, 50), lineType=cv2.LINE_AA)
            print('clicked i:{} j:{} k:{}'.format(x, y, self.dicom_processor.calc_k_on_ijk_coordinates(self.dicom_index)))
            self.ijk_eso_centers[self.dicom_index] = np.array([x, y, self.dicom_processor.calc_k_on_ijk_coordinates(self.dicom_index)])
        elif event is cv2.EVENT_RBUTTONUP and self.Is_zoomed is False:
            self.zoom_function(x, y)
        elif event is cv2.EVENT_RBUTTONUP and self.Is_zoomed is True:
            self.CT_img_for_show = self.cur_src_CT_img
            self.Is_zoomed = False

    def calc_non_zoomed_ij(self, i, j, zoom_i_ratio, zoom_j_ratio):
        non_zoomed_i, non_zoomed_j = (self.zoom_point[0] + (i / zoom_i_ratio),
                                      self.zoom_point[1] + (j / zoom_j_ratio))
        return non_zoomed_i, non_zoomed_j

    def zoom_function(self, x, y):
        self.zoom_point = [int(x - ((self.zoomed_CT_size[0] - 1) / 2)), int(y - ((self.zoomed_CT_size[1] - 1) / 2))]
        if self.zoom_point[0] < 0:
            self.zoom_point[0] = 0
        if self.src_CT_size[1] <= 0:
            self.zoom_point[1] = 0
        tmp = self.cur_src_CT_img[self.zoom_point[1]:(self.zoom_point[1] + self.zoomed_CT_size[1]),
                                  self.zoom_point[0]:(self.zoom_point[0] + self.zoomed_CT_size[0]), :]
        self.CT_img_for_show = cv2.resize(tmp, self.src_CT_size[:2], interpolation=cv2.INTER_LANCZOS4)
        self.Is_zoomed = True

    def key_function(self):
        key = cv2.waitKey(50)
        if key is ord('d') and self.dicom_index != (self.dicom_size - 1):
            self.change_cur_CT(GUI_LOAD_NEXT)
        elif key is ord('a') and self.dicom_index != 0:
            self.change_cur_CT(GUI_LOAD_PREV)

    def run(self):
        cv2.setMouseCallback('menu', self.menu_callbacks)
        cv2.setMouseCallback('CT image', self.CT_image_callbacks)

        while True:
            self.key_function()
            self.GUI_imshow()
            if not self.Is_continue:
                break