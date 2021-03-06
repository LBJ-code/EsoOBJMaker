import pydicom
import glob
import sys, os
import numpy as np
import cv2
import utils


class DicomProcessor:

    def __init__(self, dicom_dir, args):
        self.args = args
        self.dicom_dir = dicom_dir
        self.dicom_file_list = glob.glob(os.path.join(dicom_dir, '*'))
        self.dicom_size = len(self.dicom_file_list)

        if self.dicom_size == 0:
            print("dicom file has no files")
            raise

        sample_dfile = pydicom.read_file(self.dicom_file_list[0])
        self.dicom_row_num = sample_dfile.Rows
        self.dicom_col_num = sample_dfile.Columns
        self.pixel_spacing_dx, self.pixel_spacing_dy = sample_dfile.PixelSpacing
        self.slice_thickness = sample_dfile.SliceThickness
        self.image_position_patient = sample_dfile.ImagePositionPatient
        #なぜか3DSlicer上ではSが-0.25されている
        self.image_position_patient[2] -= 2.5
        self.CT_for_imshow = self.load_initial_CT()
        self.original_eso_radius = args.original_eso_radius
        self.target_eso_radius = int(args.eso_radius / self.pixel_spacing_dx)

    def calc_ijk2LPS_mat(self, dst=None):
        ijk_samples = self.calc_ijk_samples()
        LPS_samples = self.calc_LPS_samples()

        Ls = LPS_samples[:, 0]
        Ps = LPS_samples[:, 1]
        Ss = LPS_samples[:, 2]

        row1 = np.linalg.solve(ijk_samples, Ls)
        row2 = np.linalg.solve(ijk_samples, Ps)
        row3 = np.linalg.solve(ijk_samples, Ss)
        row4 = np.array([0, 0, 0, 1])

        IJKtoLPS_mat = np.array([row1, row2, row3, row4])

        if dst != None:
            dst = IJKtoLPS_mat

        return IJKtoLPS_mat

    def calc_ijk_samples(self):
        origin_k = len(self.dicom_file_list) - 1
        ijk = np.array([[0, 0, origin_k, 1],
                        [1, 0, origin_k, 1],
                        [0, 1, origin_k, 1],
                        [0, 0, origin_k - 1, 1]])

        return ijk

    def calc_LPS_samples(self):
        origin_L, origin_P, origin_S = self.image_position_patient

        LPS = np.array([[origin_L, origin_P, origin_S],
                        [origin_L + self.pixel_spacing_dx, origin_P, origin_S],
                        [origin_L, origin_P + self.pixel_spacing_dy, origin_S],
                        [origin_L, origin_P, origin_S - self.slice_thickness]])

        return LPS

    def load_initial_CT(self):
        CT_img = pydicom.read_file(self.dicom_file_list[0]).pixel_array

        return self.row2uint8(CT_img)

    def row2uint8(self, CT_row, delete_0s=True):
        if delete_0s:
            CT_row = np.where(CT_row == 0, 0, CT_row - np.min(CT_row[CT_row != 0]))

        tmp = np.array(255 * (CT_row / np.max(CT_row)), dtype=np.uint8)
        CT_img = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

        return CT_img

    def get_CT_by_index(self, dicom_index):
        CT_row = pydicom.read_file(self.dicom_file_list[dicom_index]).pixel_array

        return self.row2uint8(CT_row)

    def calc_k_on_ijk_coordinates(self, dicom_index):
        return self.dicom_size - dicom_index - 1

    def make_distorted_dicom(self, ijk_eso_centers, img_eso_radius):
        pad_eso_centers = self.liner_pad_list(ijk_eso_centers)
        os.makedirs(self.args.out_dicom_dir, exist_ok=True)
        if (self.dicom_col_num % 2 != 0) and (self.dicom_row_num % 2 != 0):
            print('It may not be able to get correct dicom image. In detail, please ask to Yukiya')
            raise
        half_col_num = self.dicom_col_num / 2
        half_row_num = self.dicom_row_num / 2
        fixed_points = np.array([[half_col_num, self.dicom_row_num], [self.dicom_col_num, self.dicom_row_num],
                                 [self.dicom_col_num, half_row_num], [self.dicom_col_num, 0],
                                 [half_col_num, 0], [0, 0],
                                 [0, half_row_num], [0, self.dicom_row_num]], dtype=np.float32)
        for dicom_index, eso_center in enumerate(pad_eso_centers):
            print("a")
            file_name = os.path.basename(self.dicom_file_list[dicom_index])
            new_dicom_path = os.path.join(self.args.out_dicom_dir, file_name)
            if eso_center is None:
                ds = pydicom.dcmread(self.dicom_file_list[dicom_index])
                #d.save_as(new_dicom_path)
                #d.save_as(self.dicom_file_list[dicom_index])
                ds.save_as(new_dicom_path)
            else:
                show_CT_img = self.get_CT_by_index(dicom_index)
                cv2.imshow("src", show_CT_img)
                src_8neighbors = utils.get_8neighbors_with_radius(self.original_eso_radius, eso_center)
                target_8neighbors = utils.get_8neighbors_with_radius(self.target_eso_radius, eso_center)

                src_points = np.vstack((src_8neighbors, fixed_points)).reshape((1, -1, 2))
                target_points = np.vstack((target_8neighbors, fixed_points)).reshape((1, -1, 2))

                matches = list()
                for i in range(len(src_points[0, :, :])):
                    matches.append(cv2.DMatch(i, i, 0))

                tps = cv2.createThinPlateSplineShapeTransformer()
                tps.estimateTransformation(target_points, src_points, matches)
                ds = pydicom.read_file(self.dicom_file_list[dicom_index])
                ds.decompress()
                out = tps.warpImage(ds.pixel_array)
                show_out = tps.warpImage(show_CT_img[:, :, 0])
                cv2.imshow("distorted", show_out)

                cv2.waitKey(100)

                #pydicom.encaps.encapsulate([d.PixelData])
                ds.PixelData = out.tobytes()
                ds.save_as(new_dicom_path)


    def liner_pad_list(self, ijk_eso_centers):
        pad_center_list = []
        Is_first = True
        prev_index = 0
        for index, eso_center in enumerate(ijk_eso_centers):
            if (eso_center is not None) and Is_first:
                prev_index = index
                Is_first = False
            elif (eso_center is not None) and (Is_first is not True):
                item_distance = index - prev_index
                if item_distance != 1:
                    average_weight = 1 / item_distance
                    for new_index in range((prev_index + 1), index, 1):
                        pad_center = (ijk_eso_centers[prev_index] * (average_weight * (index - new_index)) +
                                      ijk_eso_centers[index] * (average_weight * (new_index - prev_index)))
                        pad_center_list[new_index] = pad_center
                prev_index = index
            pad_center_list.append(eso_center)
        return pad_center_list
