# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:34:24 2022

@author: r0814655
"""

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology as smh
from skimage import measure
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from skimage import exposure
from tqdm import tqdm
from matplotlib_scalebar.scalebar import ScaleBar

# path_phase = 'C:\\Belgium test Project\\New Dataset\\Panason_phase_N05_IPR80\\'
# path_intensity = 'C:\\Belgium test Project\\New Dataset\\Panason_intensity_N05_IPR90\\'
# path_save_properties = 'C:\\Belgium test Project\\New Dataset\\Region properties from py\\'
# path_save_lab = 'C:\\Belgium test Project\\New Dataset\\Mitotic label from py\\'
# path_results = 'C:\\Belgium test Project\\New Dataset\\Segmentation\\'
# path_results = 'C:\\Belgium test Project\\New Dataset\\Segmentation_not_scaled\\'
# path_results = 'C:\\Belgium test Project\\New Dataset\\Segmentation_int\\'

# New Dataset 2
path_phase = 'C:\\Belgium test Project\\New Dataset2\\Panason_phase_N05_IPR80\\'
path_intensity = 'C:\\Belgium test Project\\New Dataset2\\Panason_intensity_N05_IPR90\\'
path_save_properties = 'C:\\Belgium test Project\\New Dataset2\\Region properties from py\\'
path_save_lab = 'C:\\Belgium test Project\\New Dataset2\\Mitotic label from py\\'
path_results = 'C:\\Belgium test Project\\New Dataset2\\Segmentation\\'
# # path_results = 'C:\\Belgium test Project\\New Dataset2\\Segmentation_not_scaled\\'
# # path_results = 'C:\\Belgium test Project\\New Dataset2\\Segmentation_int\\'



en_ph = os.listdir(path_phase)
en_in = os.listdir(path_intensity)
print('AOI segmentation')
for frames in tqdm(range(len(en_ph))):
    im_in = cv2.imread(path_intensity + en_in[frames], -1)
    im_ph = cv2.imread(path_phase + en_ph[frames], -1)
    
    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.imshow(im_in, cmap = 'gray')
    # plt.title('Intensity Image',fontsize = 15)
    # plt.subplot(1,2,2)
    # plt.imshow(im_ph,cmap = 'gray')
    # plt.title('Phase Image',fontsize = 15)
    
    max_in = np.max(im_in)
    max_ph = np.max(im_ph)
    
    # Complementarily thresholding
    bw_in = np.zeros(im_in.shape)
    bw_in[im_in <= 0.7] = 1
    
    
    bw_ph_1 = np.zeros(im_in.shape)
    bw_ph_2 = np.zeros(im_in.shape)
    bw_ph_1[im_ph <= -1.5] = 1
    bw_ph_2[im_ph > 1] = 1
    
    bw_ph = np.logical_or(bw_ph_1,bw_ph_2)
    
    comb = np.logical_or(bw_in,bw_ph)
    rem_sm_comb = smh.remove_small_objects(comb,min_size = 21)
    label_pre = measure.label(rem_sm_comb)
    props = measure.regionprops_table(label_pre,properties=['area'])
    area = props['area']
    
    for a in range(len(area)):
        sub_a = area[a]
        if sub_a > 20000:
            label_pre[label_pre == a+1] = 0
            
    pre_pro = label_pre
    pre_pro[pre_pro != 0] = 1
    fill_pre = ndi.binary_fill_holes(pre_pro).astype(np.uint8)
    
        
    # plt.figure(2)
    # plt.subplot(2,2,1)
    # plt.imshow(bw_in)
    # plt.title('Binarized intensity image',fontsize = 15)
    # plt.subplot(2,2,2)
    # plt.imshow(bw_ph)
    # plt.title('Binarized phase image',fontsize = 15)
    # plt.subplot(2,2,3)
    # plt.imshow(comb)
    # plt.title('Conbine phase and intensity image',fontsize = 15)
    # plt.subplot(2,2,4)
    # plt.imshow(fill_pre)
    # plt.title('Preprocessed',fontsize = 15)
    
    kernel = np.ones((3,3),np.uint8)
    erosion  = cv2.erode(fill_pre, kernel,iterations = 1)
    distance = ndi.distance_transform_edt(erosion)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=erosion, min_distance = 20)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    watershed_tf = watershed(-distance, markers, mask=fill_pre)
    
    fill_pre_for_rgb = fill_pre
    fill_pre_for_rgb[fill_pre == 1] = 255
    for_preview_local_min = np.dstack([fill_pre_for_rgb,fill_pre_for_rgb,fill_pre_for_rgb])
    for_preview_local_min[tuple(coords.T)] = [255,0,0]
    
    im_for_plot = np.zeros(watershed_tf.shape)
    im_for_plot[watershed_tf > 0] = 1
    contours = measure.find_contours(im_for_plot, 0.8)
    
    p2_im, p98_im = np.percentile(im_ph, (2, 95))
    resc_im_ph = exposure.rescale_intensity(im_ph, in_range=(p2_im, p98_im))
    
    
    # fig, ax = plt.subplots(figsize=(20,20))
    # ax.imshow(im_ph, cmap = 'gray')
    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color = 'red')
    # plt.title('Mitosis area',fontsize = 30)
    
    # fig, ax = plt.subplots(1,2,figsize=(20,20))
    # ax.ravel()[0].imshow(im_in, cmap = 'gray')
    # for contour in contours:
    #     ax.ravel()[0].plot(contour[:, 1], contour[:, 0], linewidth=2, color = 'red')
    # ax.ravel()[0].set_title('Event detection in an intensity image',fontsize = 30)
    # ax.ravel()[0].set_xlim((3072 - 50, 4096 - 50))
    # ax.ravel()[0].set_ylim((1000 + 100, 1768 + 100))
    # ax.ravel()[0].get_xaxis().set_visible(False)
    # ax.ravel()[0].get_yaxis().set_visible(False)
    # scalebar = ScaleBar(0.7, "um", length_fraction=0.20, location='lower right',
    #                     font_properties={"size": 20})
    # ax.ravel()[0].add_artist(scalebar)
    
    # ax.ravel()[1].imshow(im_ph, cmap = 'gray')
    # for contour in contours:
    #     ax.ravel()[1].plot(contour[:, 1], contour[:, 0], linewidth=2, color = 'red')
    # ax.ravel()[1].set_title('Event detection in a phase image',fontsize = 30)
    # scalebar = ScaleBar(1, "um", length_fraction=0.20, location='lower right',
    #                     font_properties={"size": 20})
    # ax.ravel()[1].set_xlim((3072 - 50, 4096 - 50))
    # ax.ravel()[1].set_ylim((1000 + 100, 1768 + 100))
    # ax.ravel()[1].get_xaxis().set_visible(False)
    # ax.ravel()[1].get_yaxis().set_visible(False)
    # scalebar = ScaleBar(0.7, "um", length_fraction=0.20, location='lower right',
    #                     font_properties={"size": 20})
    # ax.ravel()[1].add_artist(scalebar)
    
    # if frames < 10:
    #     pre = '000'
    # if frames >=10 and frames < 100:
    #     pre = '00'
    # if frames > 100 and frames < 1000:
    #     pre = '0'
    # if frames > 1000:
    #     pre = ''
            
    # plt.savefig(path_results + 'segmentation_'+ pre + str(frames) + '.png')
    # plt.close()
    
    # plt.figure(3)
    # plt.subplot(1,3,1)
    # plt.imshow(distance)
    # plt.title('Distance transform',fontsize = 15)
    # plt.subplot(1,3,2)
    # plt.imshow(for_preview_local_min)
    # plt.title('Local min',fontsize = 15)
    # plt.subplot(1,3,3)
    # plt.imshow(watershed_tf, cmap='jet')
    # plt.title('Watershed',fontsize = 15)

# path_im_gen = 'C:\\Belgium test Project\\New Dataset\\Segmentation\\'
# path_seq_vid = 'C:\\Belgium test Project\\New Dataset\\Seq_Vid\\'
# temp_fig = os.listdir(path_im_gen)
# vid_name = path_seq_vid + 'Scaled_Phase.mp4'
# img = []
# for t in range(len(path_im_gen)):
#     img.append(cv2.imread(path_im_gen + temp_fig[t],-1))
    
# height,width,layers = img[0].shape
# video = cv2.VideoWriter(vid_name,-1,3,(width,height))

# for j in range(len(img)):
#     video.write(img[j])

# cv2.destroyAllWindows()
# video.release()
    
    # props = measure.regionprops_table(watershed_tf,im_in,properties = ['label','area',
    #                                                                     'Centroid','axis_major_length',
    #                                                                     'axis_minor_length','eccentricity',
    #                                                                     'bbox','intensity_mean',
    #                                                                     'intensity_max','intensity_min','extent'])
    
    
    # pd_props = pd.DataFrame(props)
    
    # if frames < 10:
    #     pre_save = '00'
    # if frames >= 10 and frames < 100:
    #     pre_save = '0'
    # if frames >= 100:
    #     pre_save = ''
    
    # pd_props.to_csv(path_save_properties + 'props_' + pre_save + str(frames) + '.csv',index=None)
    
    # cv2.imwrite(path_save_lab + 'lab_' + pre_save + str(frames) + '.png',watershed_tf)    


en = os.listdir(path_save_properties)

mito_marked = []
eu_marked = []
cls_marked = []

for f in tqdm(range(len(en) - 1)):
    cur_props = pd.read_csv(path_save_properties + en[f])
    nxt_props = pd.read_csv(path_save_properties + en[f+1])
    
    c0_cur = np.asarray(cur_props['Centroid-0'])
    c1_cur = np.asarray(cur_props['Centroid-1'])
    c0_nxt = np.asarray(nxt_props['Centroid-0'])
    c1_nxt = np.asarray(nxt_props['Centroid-1'])
    cur_label = np.asarray(cur_props['label'])
    nxt_label = np.asarray(nxt_props['label'])
    
    mito_subpath = []
    mito_subeu = []
    mito_sub_cls = []
    
    for e in range(len(c0_cur)):
        pwx = np.power(c0_nxt - c0_cur[e],2)
        pwy = np.power(c1_nxt - c1_cur[e],2)
        
        eu_dist = np.sqrt(pwx + pwy)
        
        mn = np.min(eu_dist)
        mn_idx = np.argmin(eu_dist)
        
        eu_dist[mn_idx] = np.Inf
        clsed = np.min(eu_dist)
        clsed_idx = np.argmin(eu_dist)
        
        if mn < 50:
            mito_subpath.append([cur_label[e],nxt_label[mn_idx]])
            mito_subeu.append([cur_label[e],nxt_label[mn_idx],mn])
            mito_sub_cls.append([cur_label[e],nxt_label[clsed_idx],clsed])
    
    mito_marked.append(mito_subpath)
    eu_marked.append(mito_subeu)
    cls_marked.append(mito_sub_cls)

full_mito_path = []
full_eu = []
full_cls = []
full_cls_ind = []

for i in range(len(mito_marked)):
    cons_path = mito_marked[i]
    eu_path = eu_marked[i]
    cls_path = cls_marked[i]
    
    if i == 0:
        for cp in range(len(cons_path)):
            
            temp_fmp = np.concatenate(([i],cons_path[cp],[i+1]),axis = 0)
            temp_eu = np.concatenate(([i],[eu_path[cp][2]],[i+1]),axis = 0)
            temp_cls = np.concatenate(([i],[cls_path[cp][2]],[i+1]),axis = 0)
            temp_cls_ind = np.concatenate(([i],cls_path[cp][0:2],[i+1]),axis = 0)
            
            full_mito_path.append(temp_fmp)
            full_eu.append(temp_eu)
            full_cls.append(temp_cls)
            full_cls_ind.append(temp_cls_ind)
    
    if i > 0:
        keep_con_idx = []
        old_len = len(full_mito_path)
        for smpc in range(old_len):
            old_mp = full_mito_path[smpc]
            old_eu = full_eu[smpc]
            old_cls = full_cls[smpc]
            old_cls_ind = full_cls_ind[smpc]
            for ssmpc in range(len(cons_path)):
                
                if old_mp[-2] == cons_path[ssmpc][0] and old_mp[-1] == i:
                    
                    temp_fmp = np.concatenate((old_mp[0:-1],[cons_path[ssmpc][1]],[i+1]),axis = 0)
                    temp_eu = np.concatenate((old_eu[0:-1],[eu_path[ssmpc][2]],[i+1]),axis = 0)
                    temp_cls = np.concatenate((old_cls[0:-1],[cls_path[ssmpc][2]],[i+1]),axis = 0)
                    temp_cls_ind = np.concatenate((old_cls_ind[0:-1],[cls_path[ssmpc][1]],[i+1]),axis = 0)
                    
                    
                    full_mito_path[smpc] = temp_fmp
                    full_eu[smpc] = temp_eu
                    full_cls[smpc] = temp_cls
                    full_cls_ind[smpc] = temp_cls_ind
                    
                    keep_con_idx.append(ssmpc)
            
        for suc in range(len(cons_path)):
            if not suc in keep_con_idx:
                
                temp_fmp = np.concatenate(([i],cons_path[suc],[i+1]),axis = 0)
                temp_eu = np.concatenate(([i],[eu_path[suc][2]],[i+1]),axis = 0)
                temp_cls = np.concatenate(([i],[cls_path[suc][2]],[i+1]),axis = 0)
                temp_cls_ind = np.concatenate(([i],cls_path[suc][0:2],[i+1]),axis = 0)
                
                full_mito_path.append(temp_fmp)
                full_eu.append(temp_eu)
                full_cls.append(temp_cls)
                full_cls_ind.append(temp_cls_ind)
                
cons_mito_path = []
cons_mito_eu = []
cons_mito_cls = []
cons_mito_cls_ind = []


for m in range(len(full_mito_path)):
    
    tar_path = full_mito_path[m]
    tar_path = tar_path[0:-1]
    tar_eu = full_eu[m]
    tar_eu = tar_eu[0:-1]
    tar_cls = full_cls[m]
    tar_cls = tar_cls[0:-1]
    tar_cls_ind = full_cls_ind[m]
    tar_cls_ind = tar_cls_ind[0:-1]
    
    path_len = len(tar_path)
    frame_start = tar_path[0]
    early_z = np.zeros(frame_start)
    # post_z = np.zeros(197 - (len(tar_path) - 1) - frame_start)
    post_z = np.zeros(206 - (len(tar_path) - 1) - frame_start)
    
    processed_path = np.concatenate((early_z,tar_path[1:len(tar_path)],post_z),axis = 0)
    processed_eu = np.concatenate((early_z,tar_eu[1:len(tar_path)],post_z),axis = 0)
    processed_cls = np.concatenate((early_z,tar_cls[1:len(tar_path)],post_z),axis = 0)
    processed_cls_ind = np.concatenate((early_z,tar_cls_ind[1:len(tar_cls_ind)],post_z),axis = 0)
    
    if path_len < 15 and path_len > 3:
        cons_mito_path.append(processed_path)
        cons_mito_eu.append(processed_eu)
        cons_mito_cls.append(processed_cls)
        cons_mito_cls_ind.append(processed_cls_ind)
        
        
# Similarity calculation

np_cons_mito_path = np.asarray(cons_mito_path)
rekeep_cons_mito_path = []

np_cons_mito_eu = np.asarray(cons_mito_eu)
rekeep_cons_mito_eu = []

np_cons_mito_cls = np.asarray(cons_mito_cls)
rekeep_cons_mito_cls = []

np_cons_mito_cls_ind = np.asarray(cons_mito_cls_ind)
rekeep_cons_mito_cls_ind = []

skip_fg = False
kept_ind = []
sim_list = []
pos_range = []
for f in range(len(cons_mito_path[0]) - 1):
    for j in range(len(cons_mito_path) - 1):
        if not skip_fg:
            if np_cons_mito_path[j,f] != 0 and np_cons_mito_path[j,f] != np_cons_mito_path[j+1,f]:
                bef = np_cons_mito_path[j,:]
                af = np_cons_mito_path[j+1,:]
                score = 0
                div = 0
                k = f
                while bef[k] != 0 and k < len(bef) - 1:
                    div = div + 1
                    if bef[k] == af[k]:
                        score = score + 1
                    k = k+1
                    
                similarity = score/div
                sim_list.append([score,div,similarity,f,j,j+1])
                
                if similarity <  0.4 and not j in kept_ind:
                    pos_range.append([f,f+div])
                    rekeep_cons_mito_path.append(cons_mito_path[j])
                    rekeep_cons_mito_eu.append(cons_mito_eu[j])
                    rekeep_cons_mito_cls.append(cons_mito_cls[j])
                    rekeep_cons_mito_cls_ind.append(cons_mito_cls_ind[j])
                    skip_fg = True
                    kept_ind.append(j)
                    
        if skip_fg:
            skip_fg = False

# Verify the path of mitosis
print('Verify the path of mitosis')
en_lab = os.listdir(path_save_lab)
path_temp_fig = 'C:\\Belgium test Project\\New Dataset\\temp fig for video\\'
path_save_vid = 'C:\\Belgium test Project\\New Dataset\\Video for labeling\\'
# path_temp_fig = 'C:\\Belgium test Project\\New Dataset2\\temp fig for video\\'
# path_save_vid = 'C:\\Belgium test Project\\New Dataset2\\Video for labeling\\'
path_sequence_array = 'C:\\Belgium test Project\\New Dataset2\\Sequence_area\\'

phe_features = []
for rp in tqdm(range(len(rekeep_cons_mito_path))): 

    view_path = rekeep_cons_mito_path[rp]
    pos_view = pos_range[rp]
    path_length = pos_view[1] - pos_view[0] + 1
    frame_range = list(range(pos_view[0],pos_view[1]+1))
    cap_view_path = view_path[view_path != 0]
    phe_area_ecc = []
    phe_mi = []
    phe_maj = []
    phe_area = []
    phe_extent = []
    phe_mu_int = []
    phe_max_int = []
    # os.makedirs(path_sequence_array + 'Sequence_' + str(rp))
    # path_save_array = path_sequence_array + 'Sequence_' + str(rp) + '//'
    for v in range(len(cap_view_path)):
        lab_im = cv2.imread(path_save_lab + en_lab[frame_range[v]], -1)
        im = cv2.imread(path_intensity + en_in[frame_range[v]],-1)
        props = pd.read_csv(path_save_properties + en[frame_range[v]])
        min_x = props['bbox-1'][cap_view_path[v]-1]
        max_x = props['bbox-3'][cap_view_path[v]-1]
        min_y = props['bbox-0'][cap_view_path[v]-1]
        max_y = props['bbox-2'][cap_view_path[v]-1]
        c_im = lab_im
        c_im[lab_im != cap_view_path[v]] = 0
        c_im[lab_im == cap_view_path[v]] = 255
        contours = measure.find_contours(c_im)
        
        phe_area_ecc.append(1 - props['eccentricity'][cap_view_path[v]-1])
        phe_mi.append(1 - props['axis_minor_length'][cap_view_path[v]-1])
        phe_maj.append(1 - props['axis_major_length'][cap_view_path[v]-1])
        phe_area.append(1 - props['area'][cap_view_path[v]-1])
        phe_extent.append(1 - props['extent'][cap_view_path[v]-1])
        phe_mu_int.append(1 - props['intensity_mean'][cap_view_path[v]-1])
        phe_max_int.append(1 - props['intensity_max'][cap_view_path[v]-1])
        
        if v < 10:
            pre = '00'
        if v >=10 and v < 100:
            pre = '0'
        if v > 100:
            pre = ''
        
        # detected_area = im[min_y - 50:max_y + 50,min_x - 50:max_x + 50]
        # np.save(path_save_array + 'array_' + pre + str(v), detected_area)
        
        # fig, ax = plt.subplots()
        # ax.imshow(im, cmap='gray')
            
        # for contour in contours:
        #     ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
        
        # ax.axis('image')
        # ax.set_xticks([])
        # ax.set_yticks([])
        
        # plt.xlim((min_x - 50,max_x + 50))
        # plt.ylim((min_y - 50,max_y + 50))
    
        # plt.show()
        
        # if v < 10:
        #     pre = '00'
        # if v >=10 and v < 100:
        #     pre = '0'
        # if v > 100:
        #     pre = ''
        # plt.savefig(path_temp_fig + 'con_img_' + pre + str(v) + '.png')
        # plt.close()
        
    # temp_fig = os.listdir(path_temp_fig)
    # if rp < 10:
    #     pre_vid = '00'
    # if rp >=10 and v < 100:
    #     pre_vid = '0'
    # if rp > 100:
    #     pre_vid = ''
    # vid_name = path_save_vid + 'Vid_' + pre + str(rp) + '.mp4'
    # img = []
    # for t in range(len(temp_fig)):
    #     img.append(cv2.imread(path_temp_fig + temp_fig[t],-1))
        
    # height,width,layers = img[0].shape
    # video = cv2.VideoWriter(vid_name,-1,1,(width,height))
    
    # for j in range(len(img)):
    #     video.write(img[j])
    
    # cv2.destroyAllWindows()
    # video.release()
    
    # for f in range(len(temp_fig)):
    #     os.remove(path_temp_fig + temp_fig[f])
    
    arr_phe_ecc = np.asarray(phe_area_ecc)
    arr_phe_mi = np.asarray(phe_mi)
    arr_phe_maj = np.asarray(phe_maj)
    arr_phe_area = np.asarray(phe_area)
    arr_phe_extent = np.asarray(phe_extent)
    arr_phe_mu_int = np.asarray(phe_mu_int)
    arr_phe_mx_int = np.asarray(phe_max_int)
    
    range_phe_ecc = np.max(arr_phe_ecc) - np.min(arr_phe_ecc)
    range_phe_mi = np.max(arr_phe_mi) - np.min(arr_phe_mi)
    range_phe_maj = np.max(arr_phe_maj) - np.min(arr_phe_maj)
    range_phe_area = np.max(arr_phe_area) - np.min(arr_phe_area)
    range_phe_extent = np.max(arr_phe_extent) - np.min(arr_phe_extent)
    range_phe_mu_int = np.max(arr_phe_mu_int) - np.min(arr_phe_mu_int)
    range_phe_mx_int = np.max(arr_phe_mx_int) - np.min(arr_phe_mx_int)
    
    std_phe_ecc = np.std(arr_phe_ecc)
    std_phe_mi = np.std(arr_phe_mi)
    std_phe_maj = np.std(arr_phe_maj)
    std_phe_area = np.std(arr_phe_area)
    std_phe_extent = np.std(arr_phe_extent)
    std_phe_mu_int = np.std(arr_phe_mu_int)
    std_phe_mx_int = np.std(arr_phe_mx_int)
    
    phe_features.append([range_phe_ecc,range_phe_mi,range_phe_maj,range_phe_area,
                        range_phe_extent,range_phe_mu_int,range_phe_mx_int,
                        std_phe_ecc,std_phe_mi,std_phe_maj,std_phe_area,std_phe_extent,
                        std_phe_mu_int,std_phe_mx_int,path_length])
    
    
arr_phe_features = np.asarray(phe_features)
        
    

# Phase image segmentation
# path_phase_props = 'C:\\Belgium test Project\\New Dataset\\Phase properties\\'
path_phase_props = 'C:\\Belgium test Project\\New Dataset2\\Phase properties\\'

for frames in range(len(en_ph)):
    ph_im = cv2.imread(path_phase + en_ph[frames],-1)
    thresh = threshold_otsu(ph_im)
    bw = np.zeros(ph_im.shape)
    bw[ph_im > thresh] = 1
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    opening = opening.astype(bool)
    filt_bw = smh.remove_small_objects(opening,min_size = 500)
    
    lab = measure.label(filt_bw)
    props_ph = measure.regionprops_table(lab,ph_im,properties = ['label','area',
                                                                        'Centroid','axis_major_length',
                                                                        'axis_minor_length','eccentricity',
                                                                        'bbox','intensity_mean',
                                                                        'intensity_max','intensity_min'])
    pd_props_ph = pd.DataFrame(props_ph)
    
    if frames < 10:
        pre_save = '00'
    if frames >= 10 and frames < 100:
        pre_save = '0'
    if frames >= 100:
        pre_save = ''
        
    fig, ax = plt.subplots(1,2,figsize=(20,20))
    ax.ravel()[0].imshow(ph_im, cmap = 'gray')
    ax.ravel()[0].set_title('A phase image',fontsize = 30)
    # ax.ravel()[0].set_xlim((3072 - 50, 4096 - 50))
    # ax.ravel()[0].set_ylim((1000 + 100, 1768 + 100))
    ax.ravel()[0].get_xaxis().set_visible(False)
    ax.ravel()[0].get_yaxis().set_visible(False)
    scalebar = ScaleBar(1.3, "um", length_fraction=0.10, location='lower right',
                        font_properties={"size": 20})
    ax.ravel()[0].add_artist(scalebar)
    
    ax.ravel()[1].imshow(filt_bw, cmap = 'gray')
    ax.ravel()[1].set_title('Binary image of detected cells',fontsize = 30)
    scalebar = ScaleBar(1, "um", length_fraction=0.20, location='lower right',
                        font_properties={"size": 20})
    # ax.ravel()[1].set_xlim((3072 - 50, 4096 - 50))
    # ax.ravel()[1].set_ylim((1000 + 100, 1768 + 100))
    ax.ravel()[1].get_xaxis().set_visible(False)
    ax.ravel()[1].get_yaxis().set_visible(False)
    scalebar = ScaleBar(1.3, "um", length_fraction=0.10, location='lower right',
                        font_properties={"size": 20})
    ax.ravel()[1].add_artist(scalebar)
    
    # pd_props_ph.to_csv(path_phase_props + 'props_ph_' + pre_save + str(frames) + '.csv',index=None)


# Synce position

np_rekeep_cons_mito_cls = np.asanyarray(rekeep_cons_mito_cls)
np_rekeep_cons_mito_cls[np_rekeep_cons_mito_cls == 0] = np.Inf
detected_frame = np.argmin(np_rekeep_cons_mito_cls,axis = 1) + 1
detected_cls_dc = np.min(np_rekeep_cons_mito_cls,axis = 1)

np_rekeep_cons_mito_path = np.asanyarray(rekeep_cons_mito_path)  
rekeep_cons_mito_cls_ind = np.asanyarray(rekeep_cons_mito_cls_ind)  
np_rekeep_cons_mito_eu = np.asanyarray(rekeep_cons_mito_eu)  
np_rekeep_cons_mito_cls = np.asarray(rekeep_cons_mito_cls)

en_lab = os.listdir(path_save_lab)
# path_temp_detected_mito = 'C:\\Belgium test Project\\New Dataset\\Detected Mitotic frame\\'
path_temp_detected_mito = 'C:\\Belgium test Project\\New Dataset2\\Detected Mitotic frame\\'
path_temp_detected_array = 'C:\\Belgium test Project\\New Dataset2\\Detected Mitotic frame array\\'
path_temp_detected_marker = 'C:\\Belgium test Project\\New Dataset2\\marker frame array\\'

for rp in range(len(detected_frame)): 

    pos = detected_frame[rp] # frame
    cell = np_rekeep_cons_mito_path[rp][pos]
    
    lab_im = cv2.imread(path_save_lab + en_lab[pos], -1)
    im = cv2.imread(path_intensity + en_in[pos],-1)
    props = pd.read_csv(path_save_properties + en[pos])
    
    min_x = props['bbox-1'][cell - 1]
    max_x = props['bbox-3'][cell - 1]
    min_y = props['bbox-0'][cell - 1]
    max_y = props['bbox-2'][cell - 1]
    c_im = lab_im
    c_im[lab_im != cell] = 0
    c_im[lab_im == cell] = 255
    contours = measure.find_contours(c_im)
    
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
        
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'b', linewidth=2)
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.xlim((min_x - 50,max_x + 50))
    plt.ylim((min_y - 50,max_y + 50))

    plt.show()
    
    if rp < 10:
        pre = '00'
    if rp >=10 and rp < 100:
        pre = '0'
    if rp > 100:
        pre = ''
        
    cropped_im = im[min_y - 50:max_y + 50,min_x- 50:max_x + 50]
    cropped_marker = c_im[min_y - 50:max_y + 50,min_x- 50:max_x + 50]
    # plt.savefig(path_temp_detected_mito + 'detected_' + pre + str(rp) + '.png')   
    # plt.close()
    
    
    # detected_area = im[min_y - 50:max_y + 50,min_x - 50:max_x + 50]
    np.save(path_temp_detected_array + 'array_' + pre + str(rp), cropped_im)
    np.save(path_temp_detected_marker + 'array_' + pre + str(rp), cropped_marker)

en_ph_props = os.listdir(path_phase_props)
en = os.listdir(path_save_properties)

# paired_position = []

# for rp in range(len(detected_frame)): 
#     pos = detected_frame[rp]
#     cell = np_rekeep_cons_mito_path[rp][pos-1]
#     cell2 = np_rekeep_cons_mito_path[rp][pos]
    
#     props_c1 = pd.read_csv(path_save_properties + en[pos-1])
#     props_c2 = pd.read_csv(path_save_properties + en[pos])
    
#     props_ph_c1 = pd.read_csv(path_phase_props + en_ph_props[pos-1])
#     props_ph_c2 = pd.read_csv(path_phase_props + en_ph_props[pos])
    
#     c10i = props_c1['Centroid-0'][cell-1]
#     c11i = props_c1['Centroid-1'][cell-1]
    
#     c20i = props_c2['Centroid-0'][cell2-1]
#     c21i = props_c2['Centroid-1'][cell2-1]
    
#     # Mother cells
    
#     c10p = np.asarray(props_ph_c1['Centroid-0'])
#     c11p = np.asanyarray(props_ph_c1['Centroid-1'])
    
#     eu_c1ix = np.power(c10p - c10i,2)
#     eu_c1iy = np.power(c11p - c11i,2)
#     eu_c1 = np.sqrt(eu_c1ix + eu_c1iy)
#     ind_m_ph = np.argmin(eu_c1)
    
#     # Daughter cells
    
#     c20p = np.asarray(props_ph_c2['Centroid-0'])
#     c21p = np.asanyarray(props_ph_c2['Centroid-1'])
    
#     eu_c2ix = np.power(c20p - c20i,2)
#     eu_c2iy = np.power(c21p - c21i,2)
#     eu_c2 = np.sqrt(eu_c2ix + eu_c2iy)
#     ind_d_ph = np.argmin(eu_c2)
    
#     paired_position.append([[cell,ind_m_ph],[cell2,ind_d_ph]])
    

# # Features extraction

# column_features = ['Closed Euclidean distance from neighbor','Area of mother cell','Major axis of mother cell','Minor axis of mother cell',
#                     'Circularity of mother cell', 'Mean intensity of mother cell',
#                     'Max intensity of mother cell','Min intensity of mother cell',
#                     'Area of daughter cell','Major axis of daughter cell','Minor axis of daughter cell',
#                     'Circularity of daughter cell', 'Mean intensity of daughter cell',
#                     'Max intensity of daughter cell','Min intensity of daughter cell',
#                     'Area of mother cell (phase)','Major axis of mother cell (phase)','Minor axis of mother cell (phase)',
#                     'Circularity of mother cell (phase)', 'Mean intensity of mother cell (phase)',
#                     'Max intensity of mother cell (phase)','Min intensity of mother cell (phase)',
#                     'Area of daughter cell (phase)','Major axis of daughter cell (phase)','Minor axis of daughter cell (phase)',
#                     'Circularity of daughter cell (phase)', 'Mean intensity of daughter cell (phase)',
#                     'Max intensity of daughter cell (phase)','Min intensity of daughter cell (phase)',
#                     'Area ratio', 'Major axis ratio', 'Minor axis ratio', 'Circularity ratio','Mean intensity ratio',
#                     'Max intensity ratio','Min intensity ratio',
#                     'Area ratio (phase)', 'Major axis ratio (phase)', 'Minor axis ratio (phase)', 'Circularity ratio (phase)','Mean intensity ratio (phase)',
#                     'Max intensity ratio (phase)','Min intensity ratio (phase)',
#                     'Area diff', 'Major axis diff', 'Minor axis diff', 'Circularity diff','Mean intensity diff',
#                     'Max intensity diff','Min intensity diff',
#                     'Area diff (phase)', 'Major axis diff (phase)', 'Minor axis diff (phase)', 'Circularity diff (phase)','Mean intensity diff (phase)',
#                     'Max intensity diff (phase)','Min intensity diff (phase)',
#                     'Minimum of Euclidean distance','Maximum of Euclidean distance','Variance of Euclidean distance',
#                     'Mean of Euclidean distance','Range of Euclidean distance',
#                     'Minimum of daughter Euclidean distance','Maximum of daughter Euclidean distance','Variance of daughter Euclidean distance',
#                     'Mean of daughter Euclidean distance','Range of daughter Euclidean distance',
#                     'Range of circularity','Range of major axis','Range of minor axis','Range of area',
#                     'Range of extent','Range of mean intensity','Range of max intensity',
#                     'SD of circularity','SD of major axis','SD of minor axis','SD of area',
#                     'SD of extent','SD of mean intensity','SD of max intensity','Path length']

# features_list = []
# for r in range(len(paired_position)):
#     pos = detected_frame[r]
#     dist_cls = detected_cls_dc[r]
#     props_m_i = pd.read_csv(path_save_properties + en[pos-1])
#     props_d_i = pd.read_csv(path_save_properties + en[pos])
    
#     props_m_ph = pd.read_csv(path_phase_props + en_ph_props[pos-1])
#     props_d_ph = pd.read_csv(path_phase_props + en_ph_props[pos])
    
#     cell_m_i = paired_position[r][0][0]
#     cell_d_i = paired_position[r][1][0]
    
#     cell_m_ph = paired_position[r][0][1]
#     cell_d_ph = paired_position[r][1][1]
    
#     area_m = props_m_i['area'][cell_m_i - 1]
#     maj_m = props_m_i['axis_major_length'][cell_m_i - 1]
#     mi_m = props_m_i['axis_minor_length'][cell_m_i - 1]
#     cir_m = props_m_i['eccentricity'][cell_m_i - 1]
#     mean_int_m = props_m_i['intensity_mean'][cell_m_i - 1]
#     max_int_m = props_m_i['intensity_max'][cell_m_i - 1]
#     min_int_m = props_m_i['intensity_min'][cell_m_i - 1]
    
#     area_d = props_d_i['area'][cell_d_i - 1]
#     maj_d = props_d_i['axis_major_length'][cell_d_i - 1]
#     mi_d = props_d_i['axis_minor_length'][cell_d_i - 1]
#     cir_d = props_d_i['eccentricity'][cell_d_i - 1]
#     mean_int_d = props_d_i['intensity_mean'][cell_d_i - 1]
#     max_int_d = props_d_i['intensity_max'][cell_d_i - 1]
#     min_int_d = props_d_i['intensity_min'][cell_d_i - 1]
    
#     area_m_ph = props_m_ph['area'][cell_m_ph - 1]
#     maj_m_ph = props_m_ph['axis_major_length'][cell_m_ph - 1]
#     mi_m_ph = props_m_ph['axis_minor_length'][cell_m_ph - 1]
#     cir_m_ph = props_m_ph['eccentricity'][cell_m_ph - 1]
#     mean_int_m_ph = props_m_ph['intensity_mean'][cell_m_ph - 1]
#     max_int_m_ph = props_m_ph['intensity_max'][cell_m_ph - 1]
#     min_int_m_ph = props_m_ph['intensity_min'][cell_m_ph - 1]
    
#     area_d_ph = props_d_ph['area'][cell_d_ph - 1]
#     maj_d_ph = props_d_ph['axis_major_length'][cell_d_ph - 1]
#     mi_d_ph = props_d_ph['axis_minor_length'][cell_d_ph - 1]
#     cir_d_ph = props_d_ph['eccentricity'][cell_d_ph - 1]
#     mean_int_d_ph = props_d_ph['intensity_mean'][cell_d_ph - 1]
#     max_int_d_ph = props_d_ph['intensity_max'][cell_d_ph - 1]
#     min_int_d_ph = props_d_ph['intensity_min'][cell_d_ph - 1]
    
#     # Ratio features
    
#     area_ratio = area_m/area_d
#     maj_ratio = maj_m/maj_d
#     minor_ratio = mi_m/mi_d
#     cir_ratio = cir_m/cir_d
#     mint_ratio = mean_int_m/mean_int_d
#     max_ratio = max_int_m/max_int_d
#     min_ratio = min_int_m/min_int_d
    
#     area_ratio_ph = area_m_ph/area_d_ph
#     maj_ratio_ph = maj_m_ph/maj_d_ph
#     minor_ratio_ph = mi_m_ph/mi_d_ph
#     cir_ratio_ph = cir_m_ph/cir_d_ph
#     mint_ratio_ph = mean_int_m_ph/mean_int_d_ph
#     max_ratio_ph = max_int_m_ph/max_int_d_ph
#     min_ratio_ph = min_int_m_ph/min_int_d_ph
    
#     # Diff features
    
#     area_diff = area_m - area_d
#     maj_diff = maj_m - maj_d
#     minor_diff = mi_m - mi_d
#     cir_diff = cir_m - cir_d
#     mint_diff = mean_int_m - mean_int_d
#     max_diff = max_int_m - max_int_d
#     min_diff = min_int_m - min_int_d
    
#     area_diff_ph = area_m_ph - area_d_ph
#     maj_diff_ph = maj_m_ph - maj_d_ph
#     minor_diff_ph = mi_m_ph - mi_d_ph
#     cir_diff_ph = cir_m_ph - cir_d_ph
#     mint_diff_ph = mean_int_m_ph - mean_int_d_ph
#     max_diff_ph = max_int_m_ph - max_int_d_ph
#     min_diff_ph = min_int_m_ph - min_int_d_ph
    
#     # Movement features
    
#     dist = np_rekeep_cons_mito_eu[r,:]
#     dist = dist[dist != 0]
#     mn_dist = np.min(dist)
#     mx_dist = np.max(dist)
#     var_dist = np.var(dist)
#     mu_dist = np.mean(dist)
#     range_dist = mx_dist - mn_dist
    
#     cls_dist = np_rekeep_cons_mito_cls[r,:]
#     cls_dist = cls_dist[cls_dist != 0]
#     mn_cls_dist = np.min(cls_dist)
#     mx_cls_dist = np.max(cls_dist)
#     var_cls_dist = np.var(cls_dist)
#     mu_cls_dist = np.mean(cls_dist)
#     range_cls_dist = mx_cls_dist - mn_cls_dist
    
#     features_list.append([dist_cls,area_m,maj_m,mi_m,cir_m,mean_int_m,max_int_m,min_int_m,
#                       area_d,maj_d,mi_d,cir_d,mean_int_d,max_int_d,min_int_d,
#                       area_m_ph,maj_m_ph,mi_m_ph,cir_m_ph,mean_int_m_ph,max_int_m_ph,min_int_m_ph,
#                       area_d_ph,maj_d_ph,mi_d_ph,cir_d_ph,mean_int_d_ph,max_int_d_ph,min_int_d_ph,
#                       area_ratio,maj_ratio,minor_ratio,cir_ratio,mint_ratio,max_ratio,min_ratio,
#                       area_ratio_ph,maj_ratio_ph,minor_ratio_ph,cir_ratio_ph,mint_ratio_ph,max_ratio_ph,min_ratio_ph,
#                       area_diff,maj_diff,minor_diff,cir_diff,mint_diff,max_diff,min_diff,
#                       area_diff_ph,maj_diff_ph,minor_diff_ph,cir_diff_ph,mint_diff_ph,max_diff_ph,min_diff_ph,
#                       mn_dist,mx_dist,var_dist,mu_dist,range_dist,
#                       mn_cls_dist,mx_cls_dist,var_cls_dist,mu_cls_dist,range_cls_dist])
    

# arr_feature_list = np.asarray(features_list)

# arr_features = np.concatenate((arr_feature_list,arr_phe_features),axis = 1)
# features = pd.DataFrame(data = arr_features,columns = column_features)
# path_save_features = 'C:\\Belgium test Project\\New Dataset\\features_py\\'
# # path_save_features = 'C:\\Belgium test Project\\New Dataset2\\Features\\'
# features.to_csv(path_save_features + 'features.csv', index=None)
    
    

# rp = 70
# view_path = rekeep_cons_mito_path[rp]
# pos_view = pos_range[rp]
# path_length = pos_view[1] - pos_view[0] + 1
# frame_range = list(range(pos_view[0],pos_view[1]+1))
# cap_view_path = view_path[view_path != 0]
# phe_area_ecc = []
# phe_mi = []
# phe_maj = []
# phe_area = []
# phe_extent = []
# phe_mu_int = []
# phe_max_int = []
# for v in range(len(cap_view_path)):
#     lab_im = cv2.imread(path_save_lab + en_lab[frame_range[v]], -1)
#     im = cv2.imread(path_intensity + en_in[frame_range[v]],-1)
#     props = pd.read_csv(path_save_properties + en[frame_range[v]])
#     min_x = props['bbox-1'][cap_view_path[v]-1]
#     max_x = props['bbox-3'][cap_view_path[v]-1]
#     min_y = props['bbox-0'][cap_view_path[v]-1]
#     max_y = props['bbox-2'][cap_view_path[v]-1]
#     c_im = lab_im
#     c_im[lab_im != cap_view_path[v]] = 0
#     c_im[lab_im == cap_view_path[v]] = 255
#     contours = measure.find_contours(c_im)
    
#     phe_area_ecc.append(1 - props['eccentricity'][cap_view_path[v]-1])
#     phe_mi.append(1 - props['axis_minor_length'][cap_view_path[v]-1])
#     phe_maj.append(1 - props['axis_major_length'][cap_view_path[v]-1])
#     phe_area.append(1 - props['area'][cap_view_path[v]-1])
#     phe_extent.append(1 - props['extent'][cap_view_path[v]-1])
#     phe_mu_int.append(1 - props['intensity_mean'][cap_view_path[v]-1])
#     phe_max_int.append(1 - props['intensity_max'][cap_view_path[v]-1])
    

    
#     fig, ax = plt.subplots()
#     ax.imshow(im, cmap='gray')
        
#     for contour in contours:
#         ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
    
#     ax.axis('image')
#     ax.set_xticks([])
#     ax.set_yticks([])
    
#     plt.xlim((min_x - 50,max_x + 50))
#     plt.ylim((min_y - 50,max_y + 50))

#     plt.show()    
    
    
    
