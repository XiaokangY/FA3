from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD, FGSM
from utils import *
from Attacks import MIM, FA3_MIM, FA3_PGD, FA3_FGSM
import torch
import warnings
from tqdm import tqdm

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('./models/VGG13.pt')
    model.train()
    f_model = PyTorchModel(model, bounds=(0, 1))

    PGD = LinfPGD(random_start=False, rel_stepsize=1.0 / 40)
    MIM = MIM(random_start=False, rel_stepsize=1.0 / 40)
    FGSM = FGSM(random_start=False)
    FA3_PGD = FA3_PGD(random_start=False, rel_stepsize=1.0 / 40)
    FA3_MIM = FA3_MIM(random_start=False, rel_stepsize=1.0 / 40)
    FA3_FGSM = FA3_FGSM(random_start=False)

    # path_result = os.path.join('./result', "    ")
    # os.makedirs(os.path.join(path_result, "audio"))
    # os.makedirs(os.path.join(path_result, "spectrum"))

    # Attacking
    Total_SR_1 = 0
    Total_SR_2 = 0
    Total_SR_3 = 0
    Total_SR_4 = 0
    Total_SR_5 = 0
    Total_SR_6 = 0

    Euclidean_distance_1 = 0.0
    Euclidean_distance_2 = 0.0
    Euclidean_distance_3 = 0.0
    Euclidean_distance_4 = 0.0
    Euclidean_distance_5 = 0.0
    Euclidean_distance_6 = 0.0

    Manhattan_distance_1 = 0.0
    Manhattan_distance_2 = 0.0
    Manhattan_distance_3 = 0.0
    Manhattan_distance_4 = 0.0
    Manhattan_distance_5 = 0.0
    Manhattan_distance_6 = 0.0

    ssim_1 = 0.0
    ssim_2 = 0.0
    ssim_3 = 0.0
    ssim_4 = 0.0
    ssim_5 = 0.0
    ssim_6 = 0.0

    snr_1 = 0.0
    snr_2 = 0.0
    snr_3 = 0.0
    snr_4 = 0.0
    snr_5 = 0.0
    snr_6 = 0.0

    for class_id in tqdm(range(10)):
        print('')
        print(f'Now attacking the {class_id} class')
        test_data = torch.load(os.path.join('./data', f'class{class_id}', 'correct_datas.pt'))
        test_data[:, :, 0:9, :] = 0.0
        test_data[:, :, 123:127, :] = 0.0
        # Datas and labels to be attacked

        attack_data1 = torch.load(os.path.join('./data', f'class{class_id}', 'correct_datas.pt'))
        attack_label = torch.load(os.path.join('./data', f'class{class_id}', 'correct_labels.pt'))
        raw_advs1, clipped_advs1, success1 = PGD(f_model, attack_data1, attack_label, epsilons=0.03)
        SR_1 = success1.float().mean(axis=-1)
        clipped_advs1[:, :, 0:9, :] = 0.0
        clipped_advs1[:, :, 123:127, :] = 0.0

        snr_PGD = calculate_snr(test_data, clipped_advs1)
        snr_1 = snr_PGD + snr_1
        Euclidean_distance_PGD = average_euclidean_distance(test_data, clipped_advs1)
        Euclidean_distance_1 = Euclidean_distance_1 + Euclidean_distance_PGD
        Manhattan_distance_PGD = calculate_average_manhattan_distance(test_data, clipped_advs1)
        Manhattan_distance_1 = Manhattan_distance_1 + Manhattan_distance_PGD
        ssim_PGD = calculate_mean_ssim(test_data, clipped_advs1)
        ssim_1 = ssim_1 + ssim_PGD


        attack_data2 = torch.load(os.path.join('./data', f'class{class_id}', 'correct_datas.pt'))
        raw_advs2, clipped_advs2, success2 = MIM(f_model, attack_data2, attack_label, epsilons=0.03)
        SR_2 = success2.float().mean(axis=-1)
        clipped_advs2[:, :, 0:9, :] = 0.0
        clipped_advs2[:, :, 123:127, :] = 0.0

        snr_MIM = calculate_snr(test_data, clipped_advs2)
        snr_2 = snr_MIM + snr_2
        Euclidean_distance_MIM = average_euclidean_distance(test_data, clipped_advs2)
        Euclidean_distance_2 = Euclidean_distance_2 + Euclidean_distance_MIM
        Manhattan_distance_MIM = calculate_average_manhattan_distance(test_data, clipped_advs2)
        Manhattan_distance_2 = Manhattan_distance_2 + Manhattan_distance_MIM
        ssim_MIM = calculate_mean_ssim(test_data, clipped_advs2)
        ssim_2 = ssim_2 + ssim_MIM

        attack_data3 = torch.load(os.path.join('./data', f'class{class_id}', 'correct_datas.pt'))
        raw_advs3, clipped_advs3, success3 = FGSM(f_model, attack_data3, attack_label, epsilons=0.03)
        SR_3 = success3.float().mean(axis=-1)
        clipped_advs3[:, :, 0:9, :] = 0.0
        clipped_advs3[:, :, 123:127, :] = 0.0

        snr_FGSM = calculate_snr(test_data, clipped_advs3)
        snr_3 = snr_FGSM + snr_3
        Euclidean_distance_FGSM = average_euclidean_distance(test_data, clipped_advs3)
        Euclidean_distance_3 = Euclidean_distance_3 + Euclidean_distance_FGSM
        Manhattan_distance_FGSM = calculate_average_manhattan_distance(test_data, clipped_advs3)
        Manhattan_distance_3 = Manhattan_distance_3 + Manhattan_distance_FGSM
        ssim_FGSM = calculate_mean_ssim(test_data, clipped_advs3)
        ssim_3 = ssim_3 + ssim_FGSM


        attack_data4 = torch.load(os.path.join('./data', f'class{class_id}', 'correct_datas.pt'))
        raw_advs4, clipped_advs4, success4 = FA3_PGD(f_model, attack_data4, attack_label, epsilons=0.08)
        SR_4 = success4.float().mean(axis=-1)
        clipped_advs4[:, :, 0:9, :] = 0.0
        clipped_advs4[:, :, 123:127, :] = 0.0
        snr_OURSPGD = calculate_snr(test_data, clipped_advs4)
        snr_4 = snr_OURSPGD + snr_4
        Euclidean_distance_OURSPGD = average_euclidean_distance(test_data, clipped_advs4)
        Euclidean_distance_4 = Euclidean_distance_4 + Euclidean_distance_OURSPGD
        Manhattan_distance_OURSPGD = calculate_average_manhattan_distance(test_data, clipped_advs4)
        Manhattan_distance_4 = Manhattan_distance_4 + Manhattan_distance_OURSPGD
        ssim_OURS_PGD = calculate_mean_ssim(test_data, clipped_advs4)
        ssim_4 = ssim_4 + ssim_OURS_PGD

        attack_data5 = torch.load(os.path.join('./data', f'class{class_id}', 'correct_datas.pt'))
        raw_advs5, clipped_advs5, success5 = FA3_MIM(f_model, attack_data5, attack_label, epsilons=0.08)
        SR_5 = success5.float().mean(axis=-1)
        clipped_advs5[:, :, 0:9, :] = 0.0
        clipped_advs5[:, :, 123:127, :] = 0.0

        snr_OURSMIM = calculate_snr(test_data, clipped_advs5)
        snr_5 = snr_OURSMIM + snr_5
        Euclidean_distance_OURSMIM = average_euclidean_distance(test_data, clipped_advs5)
        Euclidean_distance_5 = Euclidean_distance_5 + Euclidean_distance_OURSMIM
        Manhattan_distance_OURSMIM = calculate_average_manhattan_distance(test_data, clipped_advs5)
        Manhattan_distance_5 = Manhattan_distance_5 + Manhattan_distance_OURSMIM
        ssim_OURS_MIM = calculate_mean_ssim(test_data, clipped_advs5)
        ssim_5 = ssim_5 + ssim_OURS_MIM

        attack_data6 = torch.load(os.path.join('./data', f'class{class_id}', 'correct_datas.pt'))
        raw_advs6, clipped_advs6, success6 = FA3_FGSM(f_model, attack_data6, attack_label, epsilons=0.08)
        SR_6 = success6.float().mean(axis=-1)
        clipped_advs6[:, :, 0:9, :] = 0.0
        clipped_advs6[:, :, 123:127, :] = 0.0
        snr_OURSFGSM = calculate_snr(test_data, clipped_advs6)
        snr_6 = snr_OURSFGSM + snr_6
        Euclidean_distance_OURSFGSM = average_euclidean_distance(test_data, clipped_advs6)
        Euclidean_distance_6 = Euclidean_distance_6 + Euclidean_distance_OURSFGSM
        Manhattan_distance_OURSFGSM = calculate_average_manhattan_distance(test_data, clipped_advs6)
        Manhattan_distance_6 = Manhattan_distance_6 + Manhattan_distance_OURSFGSM
        ssim_OURS_FGSM = calculate_mean_ssim(test_data, clipped_advs6)
        ssim_6 = ssim_6 + ssim_OURS_FGSM

        Total_SR_1 = SR_1.item() + Total_SR_1
        Total_SR_2 = SR_2.item() + Total_SR_2
        Total_SR_3 = SR_3.item() + Total_SR_3
        Total_SR_4 = SR_4.item() + Total_SR_4
        Total_SR_5 = SR_5.item() + Total_SR_5
        Total_SR_6 = SR_6.item() + Total_SR_6

        # final_success = success1 & success2 & success3 & success4 & success5 & success6
        # test_data = test_data[final_success]
        # clipped_advs1 = clipped_advs1[final_success]
        # clipped_advs2 = clipped_advs2[final_success]
        # clipped_advs3 = clipped_advs3[final_success]
        # clipped_advs4 = clipped_advs4[final_success]
        # clipped_advs5 = clipped_advs5[final_success]
        # clipped_advs6 = clipped_advs6[final_success]
        #
        # attack_label = attack_label[final_success]
        # cnt = 0
        # for no_attack, attacked1, attacked2, attacked3, attacked4, attacked5, attacked6, original_label, in zip(
        #         test_data, clipped_advs1,
        #         clipped_advs2, clipped_advs3,
        #         clipped_advs4, clipped_advs5, clipped_advs6, attack_label):
        #     # Back to before normalization
        #     no_attack = no_attack * (0.000003814697265625 + 80) - 80
        #     attacked1 = attacked1 * (0.000003814697265625 + 80) - 80
        #     attacked2 = attacked2 * (0.000003814697265625 + 80) - 80
        #     attacked3 = attacked3 * (0.000003814697265625 + 80) - 80
        #     attacked4 = attacked4 * (0.000003814697265625 + 80) - 80
        #     attacked5 = attacked5 * (0.000003814697265625 + 80) - 80
        #     attacked6 = attacked6 * (0.000003814697265625 + 80) - 80
        #
        #     no_attack = no_attack.squeeze().cpu().numpy()
        #     attacked1 = attacked1.squeeze().cpu().numpy()
        #     attacked2 = attacked2.squeeze().cpu().numpy()
        #     attacked3 = attacked3.squeeze().cpu().numpy()
        #     attacked4 = attacked4.squeeze().cpu().numpy()
        #     attacked5 = attacked5.squeeze().cpu().numpy()
        #     attacked6 = attacked6.squeeze().cpu().numpy()
        #
        #     convert_mel_to_audio(no_attack,
        #                          os.path.join(path_result, 'audio', f'audio2-{original_label.item()}-{cnt}-raw.wav'))
        #     convert_mel_to_audio(attacked1, os.path.join(path_result, 'audio',
        #                                                  f'audio2-{original_label.item()}-{cnt}-LinfPGD.wav'))
        #     convert_mel_to_audio(attacked2,
        #                          os.path.join(path_result, 'audio', f'audio2-{original_label.item()}-{cnt}-MIM.wav'))
        #     convert_mel_to_audio(attacked3,
        #                          os.path.join(path_result, 'audio', f'audio2-{original_label.item()}-{cnt}-FGSM.wav'))
        #     convert_mel_to_audio(attacked4, os.path.join(path_result, 'audio',f'audio2-{original_label.item()}-{cnt}-OURSPGD.wav'))
        #     convert_mel_to_audio(attacked5, os.path.join(path_result, 'audio',f'audio2-{original_label.item()}-{cnt}-OURSMIM.wav'))
        #     convert_mel_to_audio(attacked6, os.path.join(path_result, 'audio',f'audio2-{original_label.item()}-{cnt}-OURSFGSM.wav'))
        #
        #     cnt = cnt + 1

    print('')
    print('')
    print('-----------------Total-----------------')
    print(
        f'LinfPGD:    Success rate: {Total_SR_1 / 10.0 * 100:.2f}% SSIM: {ssim_1 / 10:.4f}  ED: {Euclidean_distance_1 / 10:.4f} MD: {Manhattan_distance_1 / 10:.4f}  SNR: {snr_1 / 10:.4f}\n'
        f'MIM:        Success rate: {Total_SR_2 / 10.0 * 100:.2f}% SSIM: {ssim_2 / 10:.4f}  ED: {Euclidean_distance_2 / 10:.4f} MD: {Manhattan_distance_2 / 10:.4f}  SNR: {snr_2 / 10:.4f}\n'
        f'FGSM:       Success rate: {Total_SR_3 / 10.0 * 100:.2f}% SSIM: {ssim_3 / 10:.4f}  ED: {Euclidean_distance_3 / 10:.4f} MD: {Manhattan_distance_3 / 10:.4f}  SNR: {snr_3 / 10:.4f}\n'
        f'FA3_PGD:    Success rate: {Total_SR_4 / 10.0 * 100:.2f}% SSIM: {ssim_4 / 10:.4f}  ED: {Euclidean_distance_4 / 10:.4f} MD: {Manhattan_distance_4 / 10:.4f}  SNR: {snr_4 / 10:.4f}\n'
        f'FA3_MIM:    Success rate: {Total_SR_5 / 10.0 * 100:.2f}% SSIM: {ssim_5 / 10:.4f}  ED: {Euclidean_distance_5 / 10:.4f} MD: {Manhattan_distance_5 / 10:.4f}  SNR: {snr_5 / 10:.4f}\n'
        f'FA3_FGSM:   Success rate: {Total_SR_6 / 10.0 * 100:.2f}% SSIM: {ssim_6 / 10:.4f}  ED: {Euclidean_distance_6 / 10:.4f} MD: {Manhattan_distance_6 / 10:.4f}  SNR: {snr_6 / 10:.4f}\n')