
import time
#from pycuda import driver
import argparse
import os
from pathlib import Path
import numpy as np
from copy import deepcopy
device_idx=0
devices=list(map(int,os.environ["CUDA_VISIBLE_DEVICES"].split(','))) # specify which GPU(s) to be used
devices_count=len(devices)
import cv2
#import pycuda
#import pycuda.driver as cuda
import time
#cuda.init()
thres=0.5
split_size=1024
from torch.multiprocessing import Process
import multiprocessing as mp
from multiprocessing import shared_memory
import glob
images=[]

def device_assign():
    global device_idx
    device_idx=(device_idx+1)%devices_count
    return device_idx
def dist(p1,p2):
    d=0
    for i in range(len(p1)):
        d+=(int(p1[i])-int(p2[i]))**2
    return d**0.5

def find(P,c):
    if(int(P[c])==-1):
        return c
    P[c]=find(P,int(P[c]))
    return int(P[c])

def union(P,S,u0,v0,u1,v1,U0,V0,W,H,BC):
    c1,c2=u0-U0+(v0-V0)*W,u1-U0+(v1-V0)*W
    #print("edge",c1,c2)
    h1,h2=find(P,c1),find(P,c2)
    if(h1==h2):
        BC[0]+=1
    else:
        if(S[h1]>S[h2]):
            P[h2]=h1
            S[h1]+=S[h2]
        else:
            P[h1]=h2
            S[h2]+=S[h1]

def mergeu(id,P,S,u0,v0,u1,v1,U0,V0,W,H,BC):
    #print("mergeu",u0,v0,u1,v1,BC)
    d0111=dist(images[id][v0][u0],images[id][v0][u1])
    for v in range(v0,v1):
        d0010=d0111
        d0111=dist(images[id][v+1][u0],images[id][v+1][u1])
        d0110=dist(images[id][v+1][u0],images[id][v][u1])
        d0011=dist(images[id][v][u0],images[id][v+1][u1])
        # if(v==v0):
        #     print("first round step 0",P[u0-U0],P[u1-U0],P[u0-U0+W],P[u1-U0+W])
        if(d0010<=thres):union(P,S,u0,v,u1,v,U0,V0,W,H,BC)
        
        # if(v==v0):
        #     print("first round step 1",u0,v0,u1,v1,BC[0])
        if(d0110<=thres):
            union(P,S,u0,v+1,u1,v,U0,V0,W,H,BC)
            # if(v==v0):
            #     print("first round step 2",u0,v0,u1,v1,BC[0])
            d0001=dist(images[id][v][u0],images[id][v+1][u0])
            d1011=dist(images[id][v][u1],images[id][v+1][u1])
            if(d0010<=thres and d0001<=thres):BC[0]=int(BC[0])-1
            if(d0111<=thres and d1011<=thres):BC[0]=int(BC[0])-1
            # if(v==v0):
            #     print("first round step 3",u0,v0,u1,v1,BC[0])

        elif(d0011<=thres):
            union(P,S,u0,v,u1,v+1,U0,V0,W,H,BC)
            d0001=dist(images[id][v][u0],images[id][v+1][u0])
            d1011=dist(images[id][v][u1],images[id][v+1][u1])
            if(d0010<=thres and d1011<=thres):BC[0]=int(BC[0])-1
            if(d0001<=thres and d0111<=thres):BC[0]=int(BC[0])-1
        # if(v==v0):
        #     print("first round",u0,v0,u1,v1,BC[0])
    if(d0111<=thres):union(P,S,u0,v1,u1,v1,U0,V0,W,H,BC)
    #print("after mergeu",u0,v0,u1,v1,BC)
def mergev(id,P,S,u0,v0,u1,v1,U0,V0,W,H,BC):
    #print("mergev",BC)
    d1011=dist(images[id][v0][u0],images[id][v1][u0])
    for u in range(u0,u1):
        d0001=d1011
        d1011=dist(images[id][v0][u+1],images[id][v1][u+1])
        d0110=dist(images[id][v1][u],images[id][v0][u+1])
        d0011=dist(images[id][v0][u],images[id][v1][u+1])
        if(d0001<=thres):union(P,S,u,v0,u,v1,U0,V0,W,H,BC)
        # if(u==u0):
        #     print("first round step 1",u0,v0,u1,v1,BC[0])
        if(d0110<=thres):
            union(P,S,u,v1,u+1,v0,U0,V0,W,H,BC)
            d0010=dist(images[id][v0][u],images[id][v0][u+1])
            d0111=dist(images[id][v1][u],images[id][v1][u+1])
            if(d0010<=thres and d0001<=thres):BC[0]=int(BC[0])-1
            if(d0111<=thres and d1011<=thres):BC[0]=int(BC[0])-1

        elif(d0011<=thres):
            union(P,S,u,v0,u+1,v1,U0,V0,W,H,BC)
            # if(u==u0):
            #     print("first round step 2",u0,v0,u1,v1,BC[0])
            d0010=dist(images[id][v0][u],images[id][v0][u+1])
            d0111=dist(images[id][v1][u],images[id][v1][u+1])
            if(d0010<=thres and d1011<=thres):BC[0]=int(BC[0])-1
            if(d0001<=thres and d0111<=thres):BC[0]=int(BC[0])-1
            # if(u==u0):
            #     print("first round step 3",d0001,d0111,images[id][v0][u],images[id][v0][u+1],images[id][v1][u],images[id][v1][u+1],u0,v0,u1,v1,BC[0])
        # if(u==u0):
        #     print("first round",u0,v0,u1,v1,BC[0])
    if(d1011<=thres):union(P,S,u1,v0,u1,v1,U0,V0,W,H,BC)
    #print("after mergev",BC)
def check(P,S,u0,v0,u1,v1,U0,V0,W,H):
    C=0
    for u in range(u0,u1+1):
        for v in range(v0,v1+1):
            c=u-U0+(v-V0)*W
            if(P[c]==-1):
                C+=1
            if(C>1):
                return c
    return -1
def cal(id,P,S,u0,v0,u1,v1,U0,V0,W,H,BC,num_workers=1):
    #print(f"cal id={id} u0={u0} v0={v0} u1={u1} v1={v1} BC={BC[0]}")
    if(u1-u0>=split_size and num_workers>1):
        um=(u1+u0)//2
        BC2=mp.Array('i', [0])
        t0=Process(target=cal,args=(id,P,S,u0,v0,um,v1,U0,V0,W,H,BC,num_workers//2))
        t1=Process(target=cal,args=(id,P,S,um+1,v0,u1,v1,U0,V0,W,H,BC2,num_workers//2))
        t0.start()
        t1.start()
        t0.join()
        t1.join()
        BC[0]+=BC2[0]
        t=time.time()        
        mergeu(id,P,S,um,v0,um+1,v1,U0,V0,W,H,BC)
    elif(v1-v0>=split_size and num_workers>1):
        vm=(v1+v0)//2
        BC2=mp.Array('i', [0])
        t0=Process(target=cal,args=(id,P,S,u0,v0,u1,vm,U0,V0,W,H,BC,num_workers//2))
        t1=Process(target=cal,args=(id,P,S,u0,vm+1,u1,v1,U0,V0,W,H,BC2,num_workers//2))
        t0.start()
        t1.start()
        t0.join()
        t1.join()
        BC[0]+=BC2[0]
        mergev(id,P,S,u0,vm,u1,vm+1,U0,V0,W,H,BC)
    else:
        for u in range(u0,u1):
            for v in range(v0,v1):
                d0010=dist(images[id][v][u],images[id][v][u+1])
                if(d0010<=thres):union(P,S,u,v,u+1,v,U0,V0,W,H,BC)
                d0001=dist(images[id][v][u],images[id][v+1][u])
                if(d0001<=thres):union(P,S,u,v,u,v+1,U0,V0,W,H,BC)
                d0110=dist(images[id][v+1][u],images[id][v][u+1])
                d0011=dist(images[id][v][u],images[id][v+1][u+1])
                d0111=-1
                d1011=-1
                if(d0110<=thres):
                    union(P,S,u+1,v,u,v+1,U0,V0,W,H,BC)
                    d0111=dist(images[id][v+1][u],images[id][v+1][u+1])
                    d1011=dist(images[id][v][u+1],images[id][v+1][u+1])
                    if(d0010<=thres and d0001<thres):BC[0]-=1
                    if(d0111<=thres and d1011<=thres):BC[0]-=1
                elif(d0011<=thres):
                    union(P,S,u,v,u+1,v+1,U0,V0,W,H,BC)
                    d0111=dist(images[id][v+1][u],images[id][v+1][u+1])
                    d1011=dist(images[id][v][u+1],images[id][v+1][u+1])
                    if(d0010<=thres and d1011<thres):BC[0]-=1
                    if(d0111<=thres and d0001<=thres):BC[0]-=1
            if(d0111<=thres):
                union(P,S,u,v1,u+1,v1,U0,V0,W,H,BC)
        for v in range(v0,v1):
            d0001=dist(images[id][v][u1],images[id][v+1][u1])
            if(d0001<=thres):union(P,S,u1,v,u1,v+1,U0,V0,W,H,BC)
def betti(id,pool,pool_idx,u0,v0,u1,v1,num_workers=1):
    BC=mp.Array('i',[0],lock=False)
    P=-1*np.ones(((v1-v0+1)*(u1-u0+1),))
    P=mp.Array('f',P.tolist(),lock=False)
    S=np.ones(((v1-v0+1)*(u1-u0+1),))
    S=mp.Array('f',S.tolist(),lock=False)
    cal(id,P,S,u0,v0,u1,v1,u0,v0,u1-u0+1,v1-v0+1,BC,num_workers)
    pool[pool_idx]=BC[0]
def betti_pool(id,w,h,num_workers):
    Nu=len(images[id][0])//w
    Nh=len(images[id])//h
    pool=mp.Array('i',[0 for i in range(Nu*Nh)],lock=False)
    processes=[]
    c=0
    for i in range(Nu*Nh):
        u0=w*(i%Nu)
        v0=h*(i//Nu)
        #processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w,len(images[id][0])-1),min(v0+h,len(images[id])-1),1)))
        processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w-1,len(images[id][0])-1),min(v0+h-1,len(images[id])-1),(num_workers-Nu*Nh)//(Nu*Nh))))
        processes[-1].start()
        c+=1
        if(c==num_workers):
            for process in processes:
                process.join()
            processes=[]
            c=0
    pool=np.array(pool).reshape((Nh,Nu))
    return pool

def betti_conv(id,w,h,stride,num_workers):
    #print("in betti",images[id][0][0])
    Nu=((len(images[id][0])-w)//stride)+1
    Nh=((len(images[id])-h)//stride)+1
    pool=mp.Array('i',[0 for i in range(Nu*Nh)],lock=False)
    processes=[]
    c=0
    for i in range(Nu*Nh):
        u0=stride*(i%Nu)
        v0=stride*(i//Nu)
        #processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w,len(images[id][0])-1),min(v0+h,len(images[id])-1),1)))
        processes.append(Process(target=betti,args=(id,pool,i,u0,v0,min(u0+w-1,len(images[id][0])-1),min(v0+h-1,len(images[id])-1),(num_workers-Nu*Nh)//(Nu*Nh))))
        processes[-1].start()
        c+=1
        if(c==num_workers):
            for process in processes:
                process.join()
            processes=[]
            c=0
    pool=np.array(pool).reshape((Nh,Nu))
    return pool
def draw_box(img,u0,v0,w,h):
    for u in range(u0,min(u0+w,len(img[0]))):
        img[v0][u]=img[v0+1][u]=img[min(v0+h-1,len(img)-1)][u]=img[min(v0+h,len(img)-1)][u]=[255,0,0]
    for v in range(v0,min(v0+h,len(img))):
        img[v][u0]=img[v][u0+1]=img[v][min(u0+w-1,len(img[0])-1)]=img[v][min(u0+w,len(img[0])-1)]=[255,0,0]
def draw_boxes(img,pool,betti_thres):
    W=len(img[0])//pool.shape[1]
    H=len(img)//pool.shape[0]
    for i in range(pool.shape[0]):
        for j in range(pool.shape[1]):
            if(int(pool[i,j])>betti_thres):
                r=img[H*i:H*i+H,W*j:W*j+W,0]
                g=img[H*i:H*i+H,W*j:W*j+W,1]
                b=img[H*i:H*i+H,W*j:W*j+W,2]
                r=cv2.equalizeHist(r).reshape((H,W,1))
                g=cv2.equalizeHist(g).reshape((H,W,1))
                b=cv2.equalizeHist(b).reshape((H,W,1))
                img[H*i:H*i+H,W*j:W*j+W]=np.concatenate((r,g,b),axis=-1)
                #draw_box(img,W*j,H*i,W,H)
                
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--dist_thres', type=float, default=50, help='distance threshold for connection')
    parser.add_argument('--target_upper_size', type=int, default=80, help='target object typical biggest size')
    parser.add_argument('--target_lower_size', type=int, default=20, help='target object typical smallest size')
    #parser.add_argument('--conv_stride', type=int, default=40, help='object confidence threshold')
    parser.add_argument('--betti_thres', type=int, default=0, help='lower bound of betti number for consideration of importance')
    parser.add_argument('--num_workers', type=int, default=64, help='number of parallel process')
    parser.add_argument('--save_dir', type=str, default='./', help='lower bound of betti number for consideration of importance')
    args = parser.parse_args()
    print(args)
    p = str(Path(args.source).absolute())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    img_formats=['jpg','png']
    image_file = [x for x in files if x.split('.')[-1].lower() in img_formats]
    images=[cv2.imread(img) for img in image_file]
    thres=args.dist_thres
    #image=cv2.imread(args.source)
    for i in range(len(images)):
        image=images[i]
        img0=deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        image=cv2.medianBlur(image,15)
        image1=cv2.GaussianBlur(image,(5,5),0)
        image=(4*np.array(image)-3*np.array(image1))
        image=image.reshape((args.img_size,args.img_size,1))
        images[i]=image
        poolU=betti_pool(i,args.target_upper_size,args.target_upper_size,args.num_workers)
        poolL=betti_pool(i,args.target_lower_size,args.target_lower_size,args.num_workers)
        print("poolU\n",poolU)
        print("\npoolL\n",poolL)
        #print(pool20[28:32,28:32])
        M,N=poolU.shape
        m,n=poolL.shape
        poolL_U = np.transpose(poolL.reshape(M, m//M, N, n//N),axes=(0,2,1,3)).reshape(args.img_size//args.target_upper_size,args.img_size//args.target_upper_size,(m*n)//(M*N))
        poolL_U=np.sum(poolL_U,axis=-1)
        print("\npoolL_U\n",poolL_U)
        print("\npoolU-poolL_U\n",poolU-poolL_U)
        #img_pool=img0.tolist()
        img_pool=deepcopy(img0)
        print(img_pool.shape)
        draw_boxes(img_pool,poolU-poolL_U,args.betti_thres)
        print("save in",args.save_dir+'/'+image_file[i].split('/')[-1])
        cv2.imwrite(args.save_dir+'/'+image_file[i].split('/')[-1],np.array(img_pool))
        #exit()
    f=open(args.save_dir+"/args.txt",'w')
    f.write(str(args))
    f.close()
    
