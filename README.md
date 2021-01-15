
## Usage
### Data preparation
Create the 'car' dataset (ModelNet40 data will automatically be downloaded to `data/modelnet40_ply_hdf5_2048` if needed) and log directories:
```bash
mkdir log
mkdir log/baseline
python data/create_dataset_torch.py
```
Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files (provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a>) will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.

### Train 
To train a *PCRNet* model to register point clouds, use:
```bash
CUDA_VISIBLE_DEVICES=1  python main.py -o log/baseline/SSACCR1 --sampler fps  --train-pcrnet   --epochs 250  --noise_type crop -in 1024
```
### Test
To test a *PCRNet* model to register point clouds, use:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py -o log/SAMPLENET64 --pretrained   log/baseline/SSACCR_model_best.pth   --sampler fps -in 1024  --test   --noise_type crop
```

Additional options for training and evaluating can be found using `python main.py --help`.

## Acknowledgment
This code builds upon the code provided in <a href="https://github.com/itailang/SampleNet">samplenet</a>, <a href="https://github.com/hmgoforth/PointNetLK">PointNetLK</a>, <a href="https://github.com/erikwijmans/Pointnet2_PyTorch">Pointnet2_PyTorch</a> and <a href="https://github.com/unlimblue/KNN_CUDA">KNN_CUDA</a>. We thank the authors for sharing their code.

# SCANet
