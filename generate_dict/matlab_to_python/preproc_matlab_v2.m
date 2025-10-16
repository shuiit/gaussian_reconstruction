%% load hull
clear
close all
clc

exp = 'cornell'
easyWand_name = 'calibration_easyWandData_mov9.mat'
path = 'D:\Documents\Xuehao flight data\Sparse\'
movie = 9
frame_vec = [1741:2147]

%%

mov_name = sprintf('mov%d',movie)
load([path,easyWand_name])
save_path_parent =  'G:\My Drive\Research\gaussian_splatting\gaussian_splatting_input\'


%%
save_images_dir = [save_path_parent,mov_name,'_',exp,'\','images','\'];
mkdir(save_images_dir)
for cam = 1:1:4
    
    sparse_file = sprintf('\\mov%d_cam%d_sparse.mat',movie,cam)
    sp{cam} = load([path,mov_name,sparse_file])
    cam_name =sprintf('cam%d_',cam)
    im_name = [cam_name,'bg','.mat']
    bg = sp{cam}.metaData.bg;
    save([save_images_dir,im_name],'bg')

end

%%
save_path = [save_path_parent,mov_name,'_',exp,'\','images','\']
mkdir(save_path)
save_images(sp,save_path,frame_vec)


%% world axes - from coefs

for j= 1:1:4
save_path = [save_path_parent,mov_name,'_',exp,'\','camera_KRX0']
[R,K,X0,H] = decompose_dlt(easyWandData.coefs(:,j),easyWandData.rotationMatrices(:,:,j)');
camera(:,:,j) = [K,R,X0];
rotation(:,:,j) = R; 
translation(:,:,j) = X0; 
if j == 1
    K(2,3) = 800-K(2,3)
    K(2,2) = -K(2,2)
end
pmdlt{j} = [K*R,-K*R*X0];

end
plot_camera(rotation,translation,[1,0,0;0,1,0;0,0,1],'standard wand')
save(save_path,'camera');

