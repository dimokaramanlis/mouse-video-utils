import os, glob, re, cv2, pandas, shutil, subprocess
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output

def find_mp4_files(directory):
    # List to store all .mp4 file paths
    mp4_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Find all .mp4 files in current root directory
        for file in glob.glob(os.path.join(root, '*.mp4')):
            mp4_files.append(file)

    return mp4_files

def ffmpeg_compress_mp4_video(input_path, output_path, ffmpeg_path = 'C:\\ffmpeg'):
    # function to compress
    # find ffmpeg binary

    # construct the command
    command = "ffmpeg -i head1.png -i hdmiSpitting.mov -filter_complex \"[0:v][1:v] overlay=0:0\" -pix_fmt yuv420p -c:a copy output3.mov"
    # run the command
    subprocess.run(command, capture_output=True)
    return compress_success

# Function to extract date from a path
def extract_date(path):
    date_regex = re.compile(r'\d{8}')
    match = date_regex.search(path)
    return match.group(0) if match else None

def to_analyze_dlc(path):
    splitpath = os.path.split(path)
    csvpath = glob.glob(os.path.join(splitpath[0], splitpath[1][0:13] + '*_el.csv'))
    return len(csvpath)==0

def remove_big_files(path):
    # remove h5 and pickles after you're done
    picklelisting = glob.glob(path + '*.pickle')
    hdflisting    = glob.glob(path + '*.h5')
    # remove old pickles
    for ipick in range(len(picklelisting)):
        os.remove(picklelisting[ipick])
        print('removed pickle file')
    for ihdf in range(len(hdflisting)):
        os.remove(hdflisting[ihdf])
        print('removed h5 file')
        
def copy_video_locally(sourcefile, dstfolder):
    if not os.path.exists(dstfolder):
        os.mkdir(dstfolder)
    dstfile = os.path.join(dstfolder, os.path.split(sourcefile)[1])
    print("Copying file to " + dstfile )
    shutil.copyfile(sourcefile, dstfile)
    return dstfile

def get_mouse_compartment(vidpath):
    
    cap = cv2.VideoCapture(vidpath)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    
    ret, frame   = cap.read()
    (ny,nx,ncol) = frame.shape
    Nframes      = 400
    Nskip        = 75
    imarray      = np.zeros((ny,nx, Nframes),dtype = np.uint8)
    print('determining mouse side...')
    print("frames done:")
    pbar         = tqdm(total=Nframes)
    count       = 0
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True and count < Nframes*Nskip:
        if np.mod(count,Nskip)==0:
            indas = int(np.floor(count/Nskip))
            imarray[:, :, indas] = frame[:, :, 0]
            #print('frames done: ' + str(indas) + '/'+ str(Nframes), end="\r")
            #clear_output(wait=True)  # Clear previous output
            #print('frames done: ' + str(indas) + '/'+ str(Nframes)) 
            pbar.update(1)  # Update progress bar
        count+=1
      # Break the loop
      else: 
        break
     
    # When everything done, release the video capture object
    cap.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()
    
    signalframe = imarray - np.reshape(np.median(imarray,axis=2),(ny,nx,1))
    vartop      = np.var(signalframe[int(ny/2)-300:int(ny/2),:,:])
    varbottom   = np.var(signalframe[int(ny/2):int(ny/2)+300,:,:])
    istop       = vartop > varbottom
    if istop:
        y1 = 0
        y2 = int(ny/2 + 100)
        msgprint = 'mouse found at the TOP part of the video' 
    else:
        y1 = int(ny/2-100)
        y2 = ny
        msgprint = 'mouse found at the BOTTOM part of the video' 
    print(vartop)
    print(varbottom)
    print(msgprint)
    cropval = [0, nx, y1, y2]
    return cropval

def update_saved_csv(pathupdate, yadd):
    if yadd==0:
        return
    else:
        df = pandas.read_csv(pathupdate, header = None, low_memory=False)
        
        # Identify the columns to update (starting from column 2, 0-indexed)
        cols_to_update = list(range(2, len(df.columns), 3))
        
        for col in cols_to_update:
            #df[4:, col] = pandas.to_numeric(df[col][4:])
            df[col][4:] = pandas.to_numeric(df[col][4:], errors='ignore')

        # Modify the specified columns starting from row 5 (0-indexed row 4)
        df.iloc[4:, cols_to_update] += yadd
        
        # Save the updated dataframe to a new CSV file, preserving the header structure
        df.to_csv(pathupdate, index=False, header=None)

