"""Separate process for each official repo to isolate generic model/dataset imports."""
import argparse, os, sys, pickle
from common import *
from extra_motion import canonical_motion, window_layout
p=argparse.ArgumentParser()
p.add_argument('method',choices=['flowerdance','finenet'])
p.add_argument('--work',type=Path,required=True)
p.add_argument('--seed',type=int,default=3407)
p.add_argument('--check-only',action='store_true')
a=p.parse_args();w=a.work.resolve()
os.environ.update(WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',MPLBACKEND='Agg')
repo=ROOT/'vendor'/('FlowerDance' if a.method=='flowerdance' else 'FineNet')
sys.path.insert(0,str(repo));os.chdir(repo)
trusted_torch_load_compat()
import torch
if not torch.cuda.is_available():raise SystemExit('CUDA GPU required for this pilot.')
checkpoint=ROOT/'assets'/a.method/('train-3700.pt' if a.method=='flowerdance' else 'train-2000.pt')
if not checkpoint.is_file():raise SystemExit('Upload official checkpoint to: '+str(checkpoint))
if a.method=='flowerdance':
    import inference as official
    model=official.load_model(checkpoint)
else:
    import test as official
    from args import FineDance_parse_test_opt
    saved_argv=sys.argv;sys.argv=[saved_argv[0]]
    try:opt=FineDance_parse_test_opt()
    finally:sys.argv=saved_argv
    opt.checkpoint=str(checkpoint)
    model=official.EDGE(opt,'baseline',str(checkpoint));model.eval()
print('PASS:',a.method,'official model construction and strict checkpoint loading on GPU',flush=True)
if a.check_only:
    print('Sampling and visual quality have NOT been checked by this command.')
    sys.exit(0)
cfg=read(w/'protocol.json');frames=int(cfg['frames'])
checkpoint_hash=sha(checkpoint)
for n in cfg['ids']:
    # Stable per-clip seed, unaffected by test-set ordering or constructor resets.
    clip_seed=a.seed+int(n)
    seed_all(clip_seed)
    inp=w/'inputs'/n;raw=w/'raw'/a.method/str(a.seed)/n;raw.mkdir(parents=True,exist_ok=True)
    metadata=dict(method=a.method,seed=a.seed,clip_seed=clip_seed,clip=n,
                  checkpoint_sha256=checkpoint_hash,input_audio_sha256=sha(inp/'music.wav'),
                  start_frame=0,output_frames=frames,body_joints=22)
    print('Generating',a.method,n,'frames',frames,'clip seed',clip_seed,flush=True)
    if a.method=='flowerdance':
        label=read(inp/'label.json');genre=official.parse_genre(label['style2'])
        audio,tempo_audio,tempo_sr,count=official.load_audio_clip(str(inp/'music.wav'),0,frames/30)
        bpm=official.estimate_tempo(tempo_audio,tempo_sr)
        feats=official.normalize_features(official.extract_baseline_features(audio,count,bpm))
        np.save(raw/'music_features.npy',feats)
        motion_path=official.generate_motion(model,feats,genre,raw,n,21)
        metadata.update(sampler='official Euler',steps=21,condition='music+ground_truth_style2',
                        genre=label['style2'],input_label_sha256=sha(inp/'label.json'),
                        world_conversion='z-up to y-up: -90 degrees around X')
    else:
        import librosa,soundfile as sf
        count,total=window_layout(frames)
        # Official FineNet: 4-second windows, 2-second stride; no random crop.
        audio,sr=librosa.load(str(inp/'music.wav'),sr=None,mono=True)
        required=round(frames/30*sr)
        if len(audio)<required:raise ValueError('Audio too short: '+n)
        target=round(total/30*sr)
        audio=np.pad(audio[:required],(0,target-required))
        names=[];features=[]
        for i in range(count):
            path=raw/(n+'_slice'+str(i)+'.wav')
            start=round(i*2*sr);sf.write(str(path),audio[start:start+round(4*sr)],sr,subtype='PCM_16')
            feat=official.extract(str(path))
            if feat.shape!=(120,35) or not np.isfinite(feat).all():raise ValueError('Bad FineNet feature shape/values')
            names.append(str(path));features.append(feat)
        cond=np.asarray(features,dtype=np.float32);np.save(raw/'music_features.npy',cond)
        with torch.inference_mode():
            model.render_sample((None,torch.from_numpy(cond),names),'pilot',str(raw),
                                render_count=-1,mode='long',fk_out=str(raw),render=False)
        motion_path=raw/('pilot_'+n+'.pkl')
        metadata.update(sampler='official long DDIM',steps=50,condition='music',
                        raw_frames=total,audio_tail_zero_pad_frames=total-frames,
                        crop='first protocol frames after official overlap blending',
                        world_conversion='none; original FineDance y-up coordinates')
    with motion_path.open('rb') as f:motion=pickle.load(f)
    r6,root=canonical_motion(motion['smpl_poses'],motion['smpl_trans'],frames,a.method)
    dest=w/'npz'/a.method/str(a.seed)/(n+'.npz')
    save_motion(dest,r6,root,raw_motion_sha256=sha(motion_path),**metadata)
    print('Saved',dest,flush=True)
