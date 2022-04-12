# -*- encoding: utf-8 -*-
import sys
import argparse

from Src.Evaluate.evaluate import *
from Src.Model.transu import *
from Src.NetworkTrainer.network_trainer import *
from Src.DataLoader.dataloader_CRT import read_data, pre_processing


def flip_2d(input_, list_axes):
    if 'W' in list_axes:
        input_ = input_[:, :, ::-1]

    return input_


def test_time_augmentation(trainer, input_, TTA_mode):
    list_predictions = []

    for list_flip_axes in TTA_mode:
        # Do Augmentation before forward
        augmented_input = flip_2d(input_.copy(), list_flip_axes)
        augmented_input = torch.from_numpy(augmented_input.astype(np.float32))
        augmented_input = augmented_input.unsqueeze(0).to(trainer.setting.device)
        [prediction] = trainer.setting.network(augmented_input)

        # Aug back to original order
        prediction = flip_2d(np.array(prediction.cpu().data[0, :, :, :]), list_flip_axes)

        list_predictions.append(prediction[0, :, :])

    return np.mean(list_predictions, axis=0)


def inference(trainer, list_patient_dirs, save_path, do_TTA=True):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs):
            if not os.path.exists(patient_dir):
                continue
            patient_id = patient_dir.split('/')[-1]

            prediction_dose = np.zeros((128, 256, 256), np.float32)
            gt_dose = np.zeros((128, 256, 256), np.float32)
            possible_dose_mask = np.zeros((128, 256, 256), np.uint8)

            for slice_i in range(128):
                if not os.path.exists(patient_dir + '/CT_' + str(slice_i) + '.nii.gz'):
                    continue

                dict_images = read_data(patient_dir, slice_i)
                list_images = pre_processing(dict_images)

                if do_TTA:
                    TTA_mode = [[], ['W']]
                else:
                    TTA_mode = [[]]

                prediction_single_slice = test_time_augmentation(trainer, input_=list_images[0],
                                                                 TTA_mode=TTA_mode)
                prediction_dose[slice_i, :, :] = prediction_single_slice
                gt_dose[slice_i, :, :] = list_images[1][0, :, :]
                possible_dose_mask[slice_i, :, :] = list_images[2][0, :, :]

            prediction_dose[np.logical_or(possible_dose_mask < 1, prediction_dose < 0)] = 0
            prediction_dose = 60. * prediction_dose

            prediction_nii = sitk.GetImageFromArray(prediction_dose)
            prediction_nii.SetSpacing(tuple(np.array((2,2,3), dtype='float64')))
            prediction_nii.SetOrigin((0.0, 0.0, 0.0))
            if not os.path.exists(save_path + '/' + patient_id):
                os.mkdir(save_path + '/' + patient_id)
            sitk.WriteImage(prediction_nii, save_path + '/' + patient_id + '/dose.nii.gz')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../../Output/CRT/best_val_evaluation_index.pkl')
    parser.add_argument('--TTA', type=bool, default=True,
                        help='do test-time augmentation, default True')
    parser.add_argument('--GPU_id', type=int, default=[0],
                        help='GPU_id for testing (default: [0])')
    args = parser.parse_args()

    trainer = NetworkTrainer()
    trainer.setting.project_name = 'TransU'
    trainer.setting.output_dir = '../../Output/CRT'

    trainer.setting.network = Model(vis = False)

    trainer.init_trainer(ckpt_file=args.model_path,
                         list_GPU_ids=args.GPU_id,
                         only_network=True)

    print('# \n\nStart inference !')
    list_patient_dirs = ['../../Data/pt_' + str(i) for i in range(1, 120)]


    print('# \n\nStart evaluation !')
    Dose_score = get_Dose_score_and_DVH_score(prediction_dir=trainer.setting.output_dir + '/Prediction',
                                                         gt_dir='../../Data/3D_data')

    print('\n\nDose score is: ' + str(Dose_score))
