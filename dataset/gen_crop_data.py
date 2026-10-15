import os
import glob
import shutil
import random
import argparse
import cv2

def get_croped_data_per_scene(scene_dir, patch_size=128, stride=64):
    exposure_file_path = os.path.join(scene_dir, 'exposure.txt')
    ldr_file_path = sorted(glob.glob(os.path.join(scene_dir, '*.tif')))
    label_path = os.path.join(scene_dir, 'HDRImg.hdr') 
    
    # Verifica se existem imagens suficientes na pasta antes de tentar ler
    if len(ldr_file_path) < 3:
        print(f"[!] Pulando {scene_dir}: Faltam arquivos .tif (encontrados {len(ldr_file_path)})")
        return []

    ldr_0 = cv2.imread(ldr_file_path[0], cv2.IMREAD_UNCHANGED)
    ldr_1 = cv2.imread(ldr_file_path[1], cv2.IMREAD_UNCHANGED)
    ldr_2 = cv2.imread(ldr_file_path[2], cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    # PROTEÇÃO 1: Verifica se alguma das imagens originais está corrompida
    if ldr_0 is None or ldr_1 is None or ldr_2 is None or label is None:
        print(f"[!] Pulando {scene_dir}: Uma ou mais imagens originais estão corrompidas e retornaram None.")
        return []

    crop_data = []
    h, w, _ = label.shape # 1000x1500 for Kalantari17's dataset and 1500x1125 for ICCP19's dataset
    for x in range(w):
        for y in range(h):
            if x * stride + patch_size <= w and y * stride + patch_size <= h:
                crop_ldr_0 = ldr_0[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_ldr_1 = ldr_1[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_ldr_2 = ldr_2[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_label = label[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_sample = {
                    'ldr_0': crop_ldr_0, 
                    'ldr_1': crop_ldr_1, 
                    'ldr_2': crop_ldr_2, 
                    'label': crop_label,
                    'exposure_file_path': exposure_file_path
                    }
                crop_data.append(crop_sample)
    print("{} samples of scene {}.".format(len(crop_data), scene_dir))
    return crop_data

def rotate_sample(data_sample, mode=0):
    if mode == 0:
        flag = cv2.ROTATE_90_CLOCKWISE
    elif mode == 1:
        flag = cv2.ROTATE_90_COUNTERCLOCKWISE
    rotate_ldr_0 = cv2.rotate(data_sample['ldr_0'], flag)
    rotate_ldr_1 = cv2.rotate(data_sample['ldr_1'], flag)
    rotate_ldr_2 = cv2.rotate(data_sample['ldr_2'], flag)
    rotate_label = cv2.rotate(data_sample['label'], flag)
    return {
        'ldr_0': rotate_ldr_0, 
        'ldr_1': rotate_ldr_1, 
        'ldr_2': rotate_ldr_2, 
        'label': rotate_label,
        'exposure_file_path': data_sample['exposure_file_path']
        }

def flip_sample(data_sample, mode=0):
    # mode: 0 for vertical flip and 1 for horizontal flip
    flip_ldr_0 = cv2.flip(data_sample['ldr_0'], mode)
    flip_ldr_1 = cv2.flip(data_sample['ldr_1'], mode)
    flip_ldr_2 = cv2.flip(data_sample['ldr_2'], mode)
    flip_label = cv2.flip(data_sample['label'], mode)
    return {
        'ldr_0': flip_ldr_0, 
        'ldr_1': flip_ldr_1, 
        'ldr_2': flip_ldr_2, 
        'label': flip_label,
        'exposure_file_path': data_sample['exposure_file_path']
        }

def save_sample(data_sample, save_root, id):
    save_path = os.path.join(save_root, id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    shutil.copyfile(data_sample['exposure_file_path'], os.path.join(save_path, 'exposure.txt'))
    
    # Salva os arquivos no disco
    cv2.imwrite(os.path.join(save_path, '0.tif'), data_sample['ldr_0'])
    cv2.imwrite(os.path.join(save_path, '1.tif'), data_sample['ldr_1'])
    cv2.imwrite(os.path.join(save_path, '2.tif'), data_sample['ldr_2'])
    cv2.imwrite(os.path.join(save_path, 'label.hdr'), data_sample['label'])

    # PROTEÇÃO 2: Verificação de Integridade ("Deletar o que não funciona")
    check_0 = cv2.imread(os.path.join(save_path, '0.tif'), cv2.IMREAD_UNCHANGED)
    check_1 = cv2.imread(os.path.join(save_path, '1.tif'), cv2.IMREAD_UNCHANGED)
    check_2 = cv2.imread(os.path.join(save_path, '2.tif'), cv2.IMREAD_UNCHANGED)
    check_l = cv2.imread(os.path.join(save_path, 'label.hdr'), cv2.IMREAD_UNCHANGED)

    if check_0 is None or check_1 is None or check_2 is None or check_l is None:
        print(f"\n[Aviso] Falha ao gravar a amostra {id} corretamente. Deletando pasta {save_path}...")
        shutil.rmtree(save_path)
        return False # Retorna Falso avisando que deu erro
        
    return True # Retorna Verdadeiro avisando que salvou perfeitamente

def main():
    parser = argparse.ArgumentParser(description='Prepare cropped data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # AJUSTE NOS CAMINHOS
    # Pasta raiz onde ficam os datasets
    parser.add_argument("--data_root", type=str, default='/home/gnustavo/Documentos/Projetos/P-SOTA/data')
    # Nome específico da pasta que contém as cenas originais (agora é Training_Kal)
    parser.add_argument("--source_folder", type=str, default='Training_Kal')
    
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--aug", action='store_true', default=True)

    args = parser.parse_args()

    # Caminho onde ele vai buscar as pastas (ex: .../data/Training_Kal)
    full_size_training_data_path = os.path.join(args.data_root, args.source_folder)
    
    # Caminho onde ele vai salvar as imagens recortadas (ex: .../data/sig17_training_crop128_stride64)
    cropped_training_data_path = os.path.join(args.data_root, 'sig17_training_crop{}_stride{}'.format(str(args.patch_size), str(args.stride)))
    
    if not os.path.exists(cropped_training_data_path):
        os.makedirs(cropped_training_data_path)

    global counter
    counter = 0
    
    # Busca todas as subpastas dentro de Training_Kal
    all_scenes = sorted(glob.glob(os.path.join(full_size_training_data_path, '*')))
    
    if len(all_scenes) == 0:
        print(f"Erro: Nenhuma cena encontrada em {full_size_training_data_path}")
        return

    for scene in all_scenes:
        # Pula se não for diretório
        if not os.path.isdir(scene):
            continue
            
        print("==> Processando cena: {}".format(scene))
        scene_dir = scene
        croped_data = get_croped_data_per_scene(scene_dir, patch_size=args.patch_size, stride=args.stride)
        
        for data in croped_data:
            # Só incrementa o contador se a amostra foi salva E validada com sucesso
            if save_sample(data, cropped_training_data_path, str(counter).zfill(6)):
                counter += 1

            if args.aug:
                rotate_sample_0 = rotate_sample(data, 0)
                if save_sample(rotate_sample_0, cropped_training_data_path, str(counter).zfill(6)):
                    counter += 1

                flip_sample_0 = flip_sample(data, 0)
                if save_sample(flip_sample_0, cropped_training_data_path, str(counter).zfill(6)):
                    counter += 1

    print(f"\nExtração concluída com sucesso! Total de {counter} amostras 100% válidas salvas em: {cropped_training_data_path}")

if __name__ == '__main__':
    main()