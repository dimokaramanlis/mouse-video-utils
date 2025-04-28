import os, glob, re, cv2, pandas, shutil, subprocess, time
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
#################################################################################
def find_mp4_files(directory):
    # List to store all .mp4 file paths
    mp4_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Find all .mp4 files in current root directory
        for file in glob.glob(os.path.join(root, '*.mp4')):
            mp4_files.append(file)

    return mp4_files
#################################################################################
def ffmpeg_compress_mp4_video(fvidpath, procfolder, ffmpeg_path="C:\\ffmpeg\\bin"):
    """
    Compresses an MP4 file using ffmpeg and saves it with the name 
    originalfile_reduced.mp4 in the specified destination folder.

    Args:
      fvidpath: Path to the input MP4 file.
      procfolder: Path to the destination folder.
      ffmpeg_path: Path to the ffmpeg binary (optional).
    """

    if not os.path.exists(procfolder):
        os.mkdir(procfolder)

    print(f'Encoding {fvidpath}...')

    # Extract original filename and construct new filename
    namestr, endstr = os.path.splitext(os.path.basename(fvidpath))
    foutpath = os.path.join(procfolder, f'{namestr}_reduced{endstr}')

    # Delete existing output file
    if os.path.exists(foutpath):
        os.remove(foutpath)

    # Construct the ffmpeg command
    commandStr = f'{os.path.join(ffmpeg_path, "ffmpeg")} -i {fvidpath} -c:v libx264 -crf 24 -preset veryfast -tune fastdecode -c:a copy {foutpath}'

    # Start the timer
    start_time = time.time()

    # Run the command
    status = subprocess.call(commandStr, shell=True)

    if status == 0:
        # Get file sizes
        cratio = os.stat(fvidpath).st_size / os.stat(foutpath).st_size

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Print statements for time elapsed and compression ratio
        print(f'Time elapsed: {elapsed_time:.2f} s')
        print(f'Compression ratio = {cratio:.3f}')
        
        pathfin = os.path.join(os.path.dirname(fvidpath), f'{namestr}_reduced{endstr}')
        
        # Video duration comparison using OpenCV (cv2)
        vin = cv2.VideoCapture(fvidpath)
        vindur = vin.get(cv2.CAP_PROP_POS_MSEC) / 1000
        vin.release()

        vout = cv2.VideoCapture(foutpath)
        voutdur = vout.get(cv2.CAP_PROP_POS_MSEC) / 1000
        vout.release()

        if abs(vindur - voutdur) < 1e-3:
            shutil.copyfile(foutpath, pathfin)
            os.remove(fvidpath)  # Remove the original file
        else:
            raise ValueError('Something is off with the server?')

    else:
        print('Warning: Something went wrong with compression')

    return foutpath, status
#################################################################################
# Function to extract date from a path
def extract_date(path):
    date_regex = re.compile(r'\d{8}')
    match = date_regex.search(path)
    return match.group(0) if match else None
#################################################################################
def to_analyze_dlc(path):
    splitpath = os.path.split(path)
    csvpath = glob.glob(os.path.join(splitpath[0], splitpath[1][0:13] + '*_el.csv'))
    return len(csvpath)==0
#################################################################################
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
#################################################################################        
def copy_video_locally(sourcefile, dstfolder):
    if not os.path.exists(dstfolder):
        os.mkdir(dstfolder)
    dstfile = os.path.join(dstfolder, os.path.split(sourcefile)[1])
    print("Copying file to " + dstfile )
    shutil.copyfile(sourcefile, dstfile)
    return dstfile
#################################################################################
def get_mouse_compartment(vidpath, is_slider = False):
    
    cap = cv2.VideoCapture(vidpath)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")
    
    ret, frame   = cap.read()
    (ny,nx,ncol) = frame.shape
    if is_slider:
        #------------------------------------------------------------------------
        print('cropping for slider, assering slider is on TOP...')
        y1      = np.round(ny * 0.15)
        y2      = np.round(ny * 0.9)
        cropval = [0, nx, y1, y2]
    else:
        #------------------------------------------------------------------------
        # we have to determine the side
        Nframes      = 500
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
            y1 = np.round(ny * 0.1)
            y2 = int(ny/2 + 100)
            msgprint = 'mouse found at the TOP part of the video' 
        else:
            y1 = int(ny/2-100)
            y2 = np.round(ny * 0.9)
            msgprint = 'mouse found at the BOTTOM part of the video' 
        print(vartop)
        print(varbottom)
        print(msgprint)
        cropval = [0, nx, y1, y2]
        #------------------------------------------------------------------------
    return cropval
#################################################################################
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
#################################################################################
