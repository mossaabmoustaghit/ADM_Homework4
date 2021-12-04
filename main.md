## Homework 4 ADM - Hard Coding


### Import utils

## Question 1 - Implementing your own Shazam


```python
import numpy as np      
import matplotlib.pyplot as plt 
import scipy.io.wavfile 
import subprocess
import librosa
import librosa.display
import IPython.display as ipd
from random import randint
import numpy as np
from pathlib import Path, PurePath   
from tqdm.notebook import tqdm
```


```python
def convert_mp3_to_wav(audio:str) -> str:  
    """Convert an input MP3 audio track into a WAV file.

    Args:
        audio (str): An input audio track.

    Returns:
        [str]: WAV filename.
    """
    if audio[-3:] == "mp3":
        wav_audio = audio[:-3] + "wav"
        if not Path(wav_audio).exists():
                subprocess.check_output(f"ffmpeg -i {audio} {wav_audio}", shell=True)
        return wav_audio
    
    return audio

def plot_spectrogram_and_peaks(track:np.ndarray, sr:int, peaks:np.ndarray, onset_env:np.ndarray) -> None:
    """Plots the spectrogram and peaks 

    Args:
        track (np.ndarray): A track.
        sr (int): Aampling rate.
        peaks (np.ndarray): Indices of peaks in the track.
        onset_env (np.ndarray): Vector containing the onset strength envelope.
    """
    times = librosa.frames_to_time(np.arange(len(onset_env)),
                            sr=sr, hop_length=HOP_SIZE)

    plt.figure()
    ax = plt.subplot(2, 1, 2)
    D = librosa.stft(track)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            y_axis='log', x_axis='time')
    plt.subplot(2, 1, 1, sharex=ax)
    plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    plt.vlines(times[peaks], 0,
            onset_env.max(), color='r', alpha=0.8,
            label='Selected peaks')
    plt.legend(frameon=True, framealpha=0.8)
    plt.axis('tight')
    plt.tight_layout()
    plt.show()

def load_audio_peaks(audio, offset, duration, hop_size):
    """Load the tracks and peaks of an audio.

    Args:
        audio (string, int, pathlib.Path or file-like object): [description]
        offset (float): start reading after this time (in seconds)
        duration (float): only load up to this much audio (in seconds)
        hop_size (int): the hop_length

    Returns:
        tuple: Returns the audio time series (track) and sampling rate (sr), a vector containing the onset strength envelope
        (onset_env), and the indices of peaks in track (peaks).
    """
    try:
        track, sr = librosa.load(audio, offset=offset, duration=duration)
        onset_env = librosa.onset.onset_strength(track, sr=sr, hop_length=hop_size)
        peaks = librosa.util.peak_pick(onset_env, 10, 10, 10, 10, 0.5, 0.5)
    except Error as e:
        print('An error occurred processing ', str(audio))
        print(e)

    return track, sr, onset_env, peaks
    
```


```python
N_TRACKS = 1413
HOP_SIZE = 512
OFFSET = 1.0
DURATION = 30 # TODO: to be tuned!
THRESHOLD = 0 # TODO: to be tuned!
```


```python
data_folder = Path("./mp3s-32k/")
mp3_tracks = data_folder.glob("*/*/*.mp3")
tracks = data_folder.glob("*/*/*.wav")
```


```python
a = []
for track in tqdm(mp3_tracks, total=N_TRACKS):
    #print(track)
    
    a.append(convert_mp3_to_wav(str(track)))
```


    HBox(children=(FloatProgress(value=0.0, max=1413.0), HTML(value='')))


    
    


```python
for idx, audio in enumerate(tracks):
    if idx >= 2:
        break
    track, sr, onset_env, peaks = load_audio_peaks(audio, OFFSET, DURATION, HOP_SIZE)
    plot_spectrogram_and_peaks(track, sr, peaks, onset_env)
        
```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    


We create a dataset with song information and hash values
Then we save it to json file


```python
#This function returns only indices of peaks
def ReturnPeakValues(audio):
    track, sr, onset_env, peaks = load_audio_peaks(audio, OFFSET, DURATION, HOP_SIZE)
    return peaks 
```


```python
#This function convert the Posix path to string, then extract song information 
def GetTrackInformation(track,track_id):
    info = track.split("/")
    singer = info[1] #Singer name
    album = info[2] #Album name
    song = info[3].split(".")[0] #Song name
    return [track_id,singer,album,song,track] #Return it as single list with these information
```


```python
#First we create a dataset which includes Track information and path to the tracks
track_dataset = []
for t_id,t in enumerate(tracks):
    print(t)
    track_information = GetTrackInformation(str(t),t_id)
    track_dataset.append(track_information)

len(track_dataset) #Size of our dataset
```

This is our Minhash algorithm
Each hash vector has size of 32


```python

N = 32 #Length of hash vector
max_val = (2**32)-1  #Maximum value
prime=4294967311  #Prime number
#We create tuples with size of N which are permutation values which hash all input sets
perms = [ (randint(0,max_val), randint(0,max_val)) for i in range(N)]
#Printing permutation valeus
print(perms)

def Minhash(peaks):

  # initialize a minhash of length N with positive infinity values
  vec = [float('inf') for i in range(N)]

  for peak in peaks:

    #Making sure they are all integers
    if not isinstance(peak, int): peak = hash(peak)

    #Looping over permutations
    for perm_idx, perm_vals in enumerate(perms):
      a, b = perm_vals

      #Hashing :    (a*Peak + b) % prime
      output = (a * peak + b) % prime

      #Updating vector value if output is smaller that vector 
      if vec[perm_idx] > output:
        vec[perm_idx] = output

    #Fincal vector is the minimum hash of peaks set
  return vec
```

    [(3266692686, 1224158493), (540728655, 3975749241), (2252754771, 3128148770), (1162217416, 3503989550), (781376292, 2103481465), (2916563626, 3382009535), (1478107108, 2613990718), (1948705849, 2303682133), (3356492936, 2999772786), (3154066469, 38738778), (1073628717, 1743266616), (545920280, 3495186088), (1544103190, 2561722960), (486378326, 2929084427), (317378336, 2745866174), (4193329565, 2450925772), (1923867552, 2350830800), (411907315, 852716778), (2652229423, 260175186), (4215088125, 2887582145), (485653675, 2929874567), (2230994352, 625119915), (535539238, 1064742413), (3282531942, 3282025927), (4280839747, 1065234579), (196288275, 3558002856), (1675685671, 4091614429), (231645356, 2814876136), (394972001, 449248668), (4001214944, 273211858), (3937141075, 1682725200), (3688772808, 3416121681)]
    


```python
audio_signatures = {} #A dict which keys are file id and values are signature of each song
for e in tqdm(track_dataset):
    peaks = ReturnPeakValues(e[-1]) #Getting peak indices
    song_signature = Minhash(peaks) #Minhashing each song
    audio_signatures[e[0]] = song_signature #Adding to audio_signature
  
    
    
```


    HBox(children=(FloatProgress(value=0.0, max=1413.0), HTML(value='')))


    
    


```python
import json
with open("signatures.json","w") as f:
    json.dump(audio_signatures, f, sort_keys=True, indent=4)
```

Jaccard Set Similarity


```python
def JaccardSimilarity(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2))) #Intersection between two sets
    union = (len(list1) + len(list2)) - intersection #Union
    return float(intersection) / union  #Similiarity

```

Query function for finding best matches


```python
def FindMatch(track,threshold):
    #Wrapper funciton for Query function
    result = []
    query_peaks = ReturnPeakValues(track) #Getting query peaks
    query_signature = Minhash(query_peaks) #Minhashing query peaks
    for i in audio_signatures.items(): #Iterating through the audio_dataset singatures (audio_singatures)
        similarity = JaccardSimilarity(query_signature,i[1]) #Getting Similarity
        if similarity >= threshold: #If similarity is equal or greater than threshold
            res = sorted(track_dataset)[i[0]]  #Finding track information by id
            result.append([similarity,res[1],res[2],res[3]]) #Appending complete information to result array
    return result #Returning array results of searching

#Query function
def Query(track,threshold): 
    query_result = FindMatch(track,threshold) #Calling FindMatch to get result
    for r in query_result:
        print("Song: {}, Album: {}, Band: {} , Similarity: {}".format(r[-1],r[-2],r[1],r[0])) #Printing result

#ThresholdSearching is exactly Query function with more information printing
def ThresholdSearching(track,threshold):
    query_result = FindMatch(track,threshold)
    for r in query_result:
        print("Requested track: {}, Threshold:{} ,Song: {}, Album: {}, Band: {} , Similarity: {:.4f}".format(track,threshold,r[-1],r[-2],r[1],r[0]))
```

Search for 10 requested queries


```python
queries = ["track{}.wav".format(x) for x in range(1,11)] #Making query names list (Fastest way :-D)
queries #The list of requested songs pathes
```




    ['track1.wav',
     'track2.wav',
     'track3.wav',
     'track4.wav',
     'track5.wav',
     'track6.wav',
     'track7.wav',
     'track8.wav',
     'track9.wav',
     'track10.wav']




```python
[Query(q,1.0) for q in queries] #Find songs names with 1.0 similarity
```

    Song: 03-Dream_On, Album: Aerosmith, Band: aerosmith , Similarity: 1.0
    Song: 06-I_Want_To_Break_Free, Album: The_Works, Band: queen , Similarity: 1.0
    Song: 07-October, Album: October, Band: u2 , Similarity: 1.0
    Song: 04-Ob-La-Di_Ob-La-Da, Album: The_White_Album_Disc_1, Band: beatles , Similarity: 1.0
    Song: 06-Karma_Police, Album: OK_Computer, Band: radiohead , Similarity: 1.0
    Song: 05-Heartbreaker, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 1.0
    Song: 05-Go_Your_Own_Way, Album: Rumours, Band: fleetwood_mac , Similarity: 1.0
    Song: 01-American_Idiot, Album: American_Idiot, Band: green_day , Similarity: 1.0
    Song: 06-Somebody, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 1.0
    Song: 01-Black_Friday, Album: Katy_Lied, Band: steely_dan , Similarity: 1.0
    




    [None, None, None, None, None, None, None, None, None, None]



Trying with different threshold`


```python
thresholds = [0.05, 0.1, 0.25, 0.3] #A list of thresholds to try
```


```python
for q in queries:
    for threshold in thresholds:
        ThresholdSearching(q,threshold)
```

    Requested track: track1.wav, Threshold:0.05 ,Song: 04-Blue_Jay_Way, Album: Magical_Mystery_Tour, Band: beatles , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 01-Magical_Mystery_Tour, Album: Magical_Mystery_Tour, Band: beatles , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 09-Penny_Lane, Album: Magical_Mystery_Tour, Band: beatles , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 08-Let_You_Down, Album: Crash, Band: dave_matthews_band , Similarity: 0.0847
    Requested track: track1.wav, Threshold:0.05 ,Song: 10-Sleep_To_Dream_Her, Album: Everyday, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 09-Jimi_Thing, Album: Under_The_Table_And_Dreaming, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 03-Don_t_Take_Me_Alive, Album: The_Royal_Scam, Band: steely_dan , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 09-Poor_Twisted_Me, Album: Load, Band: metallica , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 06-Dont_Tread_on_Me, Album: Metallica, Band: metallica , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 01-Fuel, Album: Reload, Band: metallica , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 03-Dream_On, Album: Aerosmith, Band: aerosmith , Similarity: 1.0000
    Requested track: track1.wav, Threshold:0.05 ,Song: 01-Cracking, Album: Suzanne_Vega, Band: suzanne_vega , Similarity: 0.1034
    Requested track: track1.wav, Threshold:0.05 ,Song: 05-Blood_Sings, Album: 99_9_F_, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 10-No_One_Knows, Album: Kerplunk, Band: green_day , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 11-Minority, Album: Warning, Band: green_day , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 10-Brain_Stew, Album: Insomniac, Band: green_day , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 04-Flick_Of_The_Wrist, Album: Sheer_Heart_Attack, Band: queen , Similarity: 0.0847
    Requested track: track1.wav, Threshold:0.05 ,Song: 04-Till_Death_Do_Us_Part, Album: Like_A_Prayer, Band: madonna , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 07-Sky_Fits_Heaven, Album: Ray_of_Light, Band: madonna , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 05-Girls_Boys, Album: Parade, Band: prince , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 03-Sail_to_the_Moon, Album: Hail_to_the_Theif, Band: radiohead , Similarity: 0.1228
    Requested track: track1.wav, Threshold:0.05 ,Song: 01-We_Shall_Be_Free, Album: The_Chase, Band: garth_brooks , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 05-Rollin_, Album: Fresh_Horses_2000, Band: garth_brooks , Similarity: 0.0847
    Requested track: track1.wav, Threshold:0.05 ,Song: 10-Tear_In_Your_Hand, Album: Little_Earthquakes, Band: tori_amos , Similarity: 0.0847
    Requested track: track1.wav, Threshold:0.05 ,Song: 11-Like_Lovers_Do, Album: Pearls_of_Passion_1997_Remaster_, Band: roxette , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 06-Vulnerable, Album: Crash_Boom_Bang_, Band: roxette , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 04-Dazed_And_Confused, Album: Led_Zeppelin_I, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 02-Babyface, Album: Zooropa, Band: u2 , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 07-Some_Days_Are_Better_Than_Others, Album: Zooropa, Band: u2 , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 05-Running_to_Stand_Still, Album: The_Joshua_Tree, Band: u2 , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 01-Zoo_Station, Album: Achtung_Baby, Band: u2 , Similarity: 0.0667
    Requested track: track1.wav, Threshold:0.05 ,Song: 07-Piggy_In_The_Mirror, Album: The_Top, Band: cure , Similarity: 0.0847
    Requested track: track1.wav, Threshold:0.1 ,Song: 03-Dream_On, Album: Aerosmith, Band: aerosmith , Similarity: 1.0000
    Requested track: track1.wav, Threshold:0.1 ,Song: 01-Cracking, Album: Suzanne_Vega, Band: suzanne_vega , Similarity: 0.1034
    Requested track: track1.wav, Threshold:0.1 ,Song: 03-Sail_to_the_Moon, Album: Hail_to_the_Theif, Band: radiohead , Similarity: 0.1228
    Requested track: track1.wav, Threshold:0.25 ,Song: 03-Dream_On, Album: Aerosmith, Band: aerosmith , Similarity: 1.0000
    Requested track: track1.wav, Threshold:0.3 ,Song: 03-Dream_On, Album: Aerosmith, Band: aerosmith , Similarity: 1.0000
    Requested track: track2.wav, Threshold:0.05 ,Song: 05-Octopus_s_Garden, Album: Abbey_Road, Band: beatles , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 13-What_You_re_Doing, Album: Beatles_For_Sale, Band: beatles , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 10-Baby_You_re_A_Rich_Man, Album: Magical_Mystery_Tour, Band: beatles , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 09-Penny_Lane, Album: Magical_Mystery_Tour, Band: beatles , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 12-You_Can_t_Do_That, Album: A_Hard_Day_s_Night, Band: beatles , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 10-Louisiana_Bayou, Album: Stand_Up, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 11-Tripping_Billies, Album: Crash, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 03-Satellite, Album: Under_The_Table_And_Dreaming, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 01-Battery, Album: Master_of_Puppets, Band: metallica , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 04-Jump_In_The_Fire, Album: Kill_Em_All, Band: metallica , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 03-Motorbreath, Album: Kill_Em_All, Band: metallica , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 06-Escape, Album: Ride_The_Lightning, Band: metallica , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 10-What_It_Takes, Album: Pump, Band: aerosmith , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 10-Pimpf, Album: Music_for_the_Masses, Band: depeche_mode , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 05-Little_15, Album: Music_for_the_Masses, Band: depeche_mode , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 03-People_Are_People, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 0.0847
    Requested track: track2.wav, Threshold:0.05 ,Song: 04-In_The_Eye, Album: Solitude_Standing, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 10-Waiting, Album: Warning, Band: green_day , Similarity: 0.0847
    Requested track: track2.wav, Threshold:0.05 ,Song: 08-Panic_Song, Album: Insomniac, Band: green_day , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 06-All_The_Time, Album: Nimrod, Band: green_day , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.05 ,Song: 06-I_Want_To_Break_Free, Album: The_Works, Band: queen , Similarity: 1.0000
    Requested track: track2.wav, Threshold:0.05 ,Song: 02-Little_Red_Corvette, Album: 1999, Band: prince , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 05-Girls_Boys, Album: Parade, Band: prince , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.05 ,Song: 03-I_Wonder_U, Album: Parade, Band: prince , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 04-Under_The_Cherry_Moon, Album: Parade, Band: prince , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 02-Sexy_M_F_, Album: The_Love_Symbol_Album, Band: prince , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 07-When_the_Sun_Goes_Down, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 01-Skies_the_Limit, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 13-The_Second_Time, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 01-The_Thunder_Rolls, Album: No_Fences_The_Limited_Series_, Band: garth_brooks , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 06-Dixie_Chicken, Album: The_Chase, Band: garth_brooks , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.05 ,Song: 09-Bring_It_On_Home, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.05 ,Song: 05-Your_Time_Is_Gonna_Come, Album: Led_Zeppelin_I, Band: led_zeppelin , Similarity: 0.0847
    Requested track: track2.wav, Threshold:0.05 ,Song: 08-The_First_Time, Album: Zooropa, Band: u2 , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 08-Indian_Summer_Sky, Album: The_Unforgettable_Fire, Band: u2 , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 03-Chameleon, Album: Pendulum, Band: creedence_clearwater_revival , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 03-Travelin_Band, Album: Cosmo_s_Factory, Band: creedence_clearwater_revival , Similarity: 0.0847
    Requested track: track2.wav, Threshold:0.05 ,Song: 02-Bootleg, Album: Bayou_Country, Band: creedence_clearwater_revival , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 02-Club_America, Album: Wild_Mood_Swings, Band: cure , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.05 ,Song: 08-Round_Round_Round, Album: Wild_Mood_Swings, Band: cure , Similarity: 0.0847
    Requested track: track2.wav, Threshold:0.05 ,Song: 03-Closedown, Album: Disintegration, Band: cure , Similarity: 0.0667
    Requested track: track2.wav, Threshold:0.1 ,Song: 06-All_The_Time, Album: Nimrod, Band: green_day , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.1 ,Song: 06-I_Want_To_Break_Free, Album: The_Works, Band: queen , Similarity: 1.0000
    Requested track: track2.wav, Threshold:0.1 ,Song: 05-Girls_Boys, Album: Parade, Band: prince , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.1 ,Song: 06-Dixie_Chicken, Album: The_Chase, Band: garth_brooks , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.1 ,Song: 09-Bring_It_On_Home, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 0.1034
    Requested track: track2.wav, Threshold:0.25 ,Song: 06-I_Want_To_Break_Free, Album: The_Works, Band: queen , Similarity: 1.0000
    Requested track: track2.wav, Threshold:0.3 ,Song: 06-I_Want_To_Break_Free, Album: The_Works, Band: queen , Similarity: 1.0000
    Requested track: track3.wav, Threshold:0.05 ,Song: 02-Something, Album: Abbey_Road, Band: beatles , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 02-The_Memory_Remains, Album: Reload, Band: metallica , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 09-Big_Muff, Album: Speak_and_Spell, Band: depeche_mode , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 12-Song_Of_Sand, Album: 99_9_F_, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 05-Blood_Sings, Album: 99_9_F_, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 10-Bedtime_Story, Album: Bedtime_Stories, Band: madonna , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 06-Life_Can_be_so_Nice, Album: Parade, Band: prince , Similarity: 0.0847
    Requested track: track3.wav, Threshold:0.05 ,Song: 10-The_Continental, Album: The_Love_Symbol_Album, Band: prince , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 06-What_Makes_You_Think_You_re_the_One, Album: Tusk, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 05-Which_One_Of_Them, Album: Ropin_The_Wind_The_Limited_Series_, Band: garth_brooks , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 08-Don_t_Make_Me_Come_To_Vegas, Album: Scarlet_s_Walk, Band: tori_amos , Similarity: 0.0847
    Requested track: track3.wav, Threshold:0.05 ,Song: 06-Black_Mountain_Side, Album: Led_Zeppelin_I, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 10-The_Wanderer, Album: Zooropa, Band: u2 , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 07-October, Album: October, Band: u2 , Similarity: 1.0000
    Requested track: track3.wav, Threshold:0.05 ,Song: 06-Born_To_Move, Album: Pendulum, Band: creedence_clearwater_revival , Similarity: 0.0667
    Requested track: track3.wav, Threshold:0.05 ,Song: 07-Up_Around_The_Bend, Album: Cosmo_s_Factory, Band: creedence_clearwater_revival , Similarity: 0.0847
    Requested track: track3.wav, Threshold:0.1 ,Song: 07-October, Album: October, Band: u2 , Similarity: 1.0000
    Requested track: track3.wav, Threshold:0.25 ,Song: 07-October, Album: October, Band: u2 , Similarity: 1.0000
    Requested track: track3.wav, Threshold:0.3 ,Song: 07-October, Album: October, Band: u2 , Similarity: 1.0000
    Requested track: track4.wav, Threshold:0.05 ,Song: 10-Sun_King, Album: Abbey_Road, Band: beatles , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 04-Oh_Darling, Album: Abbey_Road, Band: beatles , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 09-You_Never_Give_Me_Your_Money, Album: Abbey_Road, Band: beatles , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 04-Ob-La-Di_Ob-La-Da, Album: The_White_Album_Disc_1, Band: beatles , Similarity: 1.0000
    Requested track: track4.wav, Threshold:0.05 ,Song: 01-Busted_Stuff, Album: Busted_Stuff, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 09-Lie_In_Our_Graves, Album: Crash, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 12-Hidden_Track_1, Album: Before_These_Crowded_Streets, Band: dave_matthews_band , Similarity: 0.0847
    Requested track: track4.wav, Threshold:0.05 ,Song: 03-Rose_Darling, Album: Katy_Lied, Band: steely_dan , Similarity: 0.0847
    Requested track: track4.wav, Threshold:0.05 ,Song: 02-Bad_Sneakers, Album: Katy_Lied, Band: steely_dan , Similarity: 0.0847
    Requested track: track4.wav, Threshold:0.05 ,Song: 11-Mama_Said, Album: Load, Band: metallica , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 07-Creeping_Death, Album: Ride_The_Lightning, Band: metallica , Similarity: 0.0847
    Requested track: track4.wav, Threshold:0.05 ,Song: 09-World_full_of_nothing, Album: Black_Celebration, Band: depeche_mode , Similarity: 0.0847
    Requested track: track4.wav, Threshold:0.05 ,Song: 13-To_have_and_to_hold_Spanish_Taster_, Album: Music_for_the_Masses, Band: depeche_mode , Similarity: 0.0847
    Requested track: track4.wav, Threshold:0.05 ,Song: 02-Lie_To_Me, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 0.1852
    Requested track: track4.wav, Threshold:0.05 ,Song: 03-_I_ll_Never_Be_Your_Maggie_May, Album: Songs_in_Red_and_Gray, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 07-86, Album: Insomniac, Band: green_day , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 18-Prosthetic_Head, Album: Nimrod, Band: green_day , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 01-Burnout, Album: Dookie, Band: green_day , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 07-Forbidden_Love, Album: Bedtime_Stories, Band: madonna , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 11-Take_A_Bow, Album: Bedtime_Stories, Band: madonna , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 09-Prove_Yourself, Album: Pablo_Honey, Band: radiohead , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 12-The_Tourist, Album: OK_Computer, Band: radiohead , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 01-Planet_Telex, Album: The_Bends, Band: radiohead , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 06-Where_I_End_and_You_Begin, Album: Hail_to_the_Theif, Band: radiohead , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 12-You_and_I_Part_II, Album: Tango_In_The_Night, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 02-Love_Is_Dangerous, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 04-Save_Me_a_Place, Album: Tusk, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 08-The_Red_Strokes, Album: In_Pieces, Band: garth_brooks , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 09-Callin_Baton_Rouge, Album: In_Pieces, Band: garth_brooks , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 04-Jamaica_Inn, Album: The_Beekeeper, Band: tori_amos , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 06-i_i_e_e_e, Album: From_the_Choirgirl_Hotel, Band: tori_amos , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 08-Dangerous, Album: Look_Sharp_, Band: roxette , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 09-Bron-Y-Aur_Stomp, Album: Led_Zeppelin_III, Band: led_zeppelin , Similarity: 0.0847
    Requested track: track4.wav, Threshold:0.05 ,Song: 01-Immigrant_Song, Album: Led_Zeppelin_III, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 02-I_Fall_Down, Album: October, Band: u2 , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.05 ,Song: 03-With_or_Without_You, Album: The_Joshua_Tree, Band: u2 , Similarity: 0.0667
    Requested track: track4.wav, Threshold:0.1 ,Song: 04-Ob-La-Di_Ob-La-Da, Album: The_White_Album_Disc_1, Band: beatles , Similarity: 1.0000
    Requested track: track4.wav, Threshold:0.1 ,Song: 02-Lie_To_Me, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 0.1852
    Requested track: track4.wav, Threshold:0.25 ,Song: 04-Ob-La-Di_Ob-La-Da, Album: The_White_Album_Disc_1, Band: beatles , Similarity: 1.0000
    Requested track: track4.wav, Threshold:0.3 ,Song: 04-Ob-La-Di_Ob-La-Da, Album: The_White_Album_Disc_1, Band: beatles , Similarity: 1.0000
    Requested track: track5.wav, Threshold:0.05 ,Song: 17-Her_Majesty, Album: Abbey_Road, Band: beatles , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 06-Mr_Moonlight, Album: Beatles_For_Sale, Band: beatles , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 07-She_Said_She_Said, Album: Revolver, Band: beatles , Similarity: 0.0847
    Requested track: track5.wav, Threshold:0.05 ,Song: 02-Old_Dirt_Hill_Bring_That_Beat_Back_, Album: Stand_Up, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 05-Wherever_I_May_Roam, Album: Metallica, Band: metallica , Similarity: 0.0847
    Requested track: track5.wav, Threshold:0.05 ,Song: 05-Mama_Kin, Album: Aerosmith, Band: aerosmith , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 03-Going_Down-Love_In_An_Elevator, Album: Pump, Band: aerosmith , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 05-SOS, Album: Get_Your_Wings, Band: aerosmith , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 02-I_Sometimes_Wish_I_Was_Dead, Album: Speak_and_Spell, Band: depeche_mode , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 03-Stuck_With_Me, Album: Insomniac, Band: green_day , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 10-Fun_It, Album: Jazz, Band: queen , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 04-Don_t_Stop, Album: Bedtime_Stories, Band: madonna , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 12-In_This_Life, Album: Erotica, Band: madonna , Similarity: 0.0847
    Requested track: track5.wav, Threshold:0.05 ,Song: 02-Swim, Album: Ray_of_Light, Band: madonna , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 09-Forever_In_My_Life, Album: Sign_O_The_Times_Disc_1_, Band: prince , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 03-The_beautiful_ones, Album: Purple_Rain, Band: prince , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 06-Karma_Police, Album: OK_Computer, Band: radiohead , Similarity: 1.0000
    Requested track: track5.wav, Threshold:0.05 ,Song: 10-Sugar_Daddy, Album: Fleetwood_Mac, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 07-Say_You_Love_Me, Album: Fleetwood_Mac, Band: fleetwood_mac , Similarity: 0.0847
    Requested track: track5.wav, Threshold:0.05 ,Song: 01-Love_in_Store, Album: Mirage, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 07-The_Beaches_of_Cheyenne, Album: Fresh_Horses_2000, Band: garth_brooks , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 09-Ribbons_Undone, Album: The_Beekeeper, Band: tori_amos , Similarity: 0.0847
    Requested track: track5.wav, Threshold:0.05 ,Song: 12-Original_Sinsuality, Album: The_Beekeeper, Band: tori_amos , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 01-Harleys_Indians_Riders_In_The_Sky_, Album: Crash_Boom_Bang_, Band: roxette , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 12-Love_Is_Blindness, Album: Achtung_Baby, Band: u2 , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 11-Homesick, Album: Disintegration, Band: cure , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 04-Six_Different_Ways, Album: The_Head_on_the_Door, Band: cure , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.05 ,Song: 06-The_Baby_Screams, Album: The_Head_on_the_Door, Band: cure , Similarity: 0.0667
    Requested track: track5.wav, Threshold:0.1 ,Song: 06-Karma_Police, Album: OK_Computer, Band: radiohead , Similarity: 1.0000
    Requested track: track5.wav, Threshold:0.25 ,Song: 06-Karma_Police, Album: OK_Computer, Band: radiohead , Similarity: 1.0000
    Requested track: track5.wav, Threshold:0.3 ,Song: 06-Karma_Police, Album: OK_Computer, Band: radiohead , Similarity: 1.0000
    Requested track: track6.wav, Threshold:0.05 ,Song: 17-Her_Majesty, Album: Abbey_Road, Band: beatles , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 07-Here_Comes_The_Sun, Album: Abbey_Road, Band: beatles , Similarity: 0.1228
    Requested track: track6.wav, Threshold:0.05 ,Song: 09-And_Your_Bird_Can_Sing, Album: Revolver, Band: beatles , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 08-Happiness_Is_A_Warm_Gun, Album: The_White_Album_Disc_1, Band: beatles , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 03-If_I_Fell, Album: A_Hard_Day_s_Night, Band: beatles , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 10-Louisiana_Bayou, Album: Stand_Up, Band: dave_matthews_band , Similarity: 0.1228
    Requested track: track6.wav, Threshold:0.05 ,Song: 07-Drive_In_Drive_Out, Album: Crash, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 11-Tripping_Billies, Album: Crash, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-So_Right, Album: Everyday, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 10-Sleep_To_Dream_Her, Album: Everyday, Band: dave_matthews_band , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 06-Halloween, Album: Before_These_Crowded_Streets, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 10-Throw_Back_The_Little_Ones, Album: Katy_Lied, Band: steely_dan , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.05 ,Song: 03-The_House_Jack_Built, Album: Load, Band: metallica , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 06-Escape, Album: Ride_The_Lightning, Band: metallica , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 06-Dulcimer_Stomp-The_Other_Side, Album: Pump, Band: aerosmith , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-Uncle_Salty, Album: Toys_In_The_Attic, Band: aerosmith , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 06-Sweet_Emotion, Album: Toys_In_The_Attic, Band: aerosmith , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 01-Toys_In_The_Attic, Album: Toys_In_The_Attic, Band: aerosmith , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 07-Seasons_Of_Wither, Album: Get_Your_Wings, Band: aerosmith , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 04-Woman_Of_The_World, Album: Get_Your_Wings, Band: aerosmith , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-Lord_Of_The_Thighs, Album: Get_Your_Wings, Band: aerosmith , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 07-The_Landscape_Is_Changing, Album: Construction_Time_Again, Band: depeche_mode , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 12-Never_let_me_down_again_Aggro_mix_, Album: Music_for_the_Masses, Band: depeche_mode , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 09-Shouldn_t_Have_Done_That, Album: A_Broken_Frame, Band: depeche_mode , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-My_Secret_Garden, Album: A_Broken_Frame, Band: depeche_mode , Similarity: 0.1228
    Requested track: track6.wav, Threshold:0.05 ,Song: 03-Ironbound-Fancy_Poultry, Album: Solitude_Standing, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Night_Vision, Album: Solitude_Standing, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-Widow_s_Walk, Album: Songs_in_Red_and_Gray, Band: suzanne_vega , Similarity: 0.1636
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-One_For_the_Razorbacks, Album: Kerplunk, Band: green_day , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 08-80, Album: Kerplunk, Band: green_day , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 06-Dominated_Love_Slave, Album: Kerplunk, Band: green_day , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 08-Hold_On, Album: Warning, Band: green_day , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 13-Tight_Wad_Hill, Album: Insomniac, Band: green_day , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 06-All_The_Time, Album: Nimrod, Band: green_day , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 04-Longview, Album: Dookie, Band: green_day , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 11-Coming_Clean, Album: Dookie, Band: green_day , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Crazy_Little_Thing_Called_Love, Album: The_Game, Band: queen , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 11-Radio_Ga_Ga_Extended_Version_, Album: The_Works, Band: queen , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 12-I_Want_To_Break_Free_Extended_Mix_, Album: The_Works, Band: queen , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 12-Don_t_Stop_Me_Now, Album: Jazz, Band: queen , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 10-Bedtime_Story, Album: Bedtime_Stories, Band: madonna , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 03-Bye_Bye_Baby, Album: Erotica, Band: madonna , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Skin, Album: Ray_of_Light, Band: madonna , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-Sexy_M_F_, Album: The_Love_Symbol_Album, Band: prince , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 09-Bullet_Proof_I_Wish_I_Was, Album: The_Bends, Band: radiohead , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 10-Like_Spinning_Plates, Album: Amnesiac, Band: radiohead , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 07-Say_You_Love_Me, Album: Fleetwood_Mac, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 13-The_Second_Time, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Everytime_That_It_Rains, Album: Garth_Brooks_The_Limited_Series_, Band: garth_brooks , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 01-Parasol, Album: The_Beekeeper, Band: tori_amos , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 14-The_Beekeeper, Album: The_Beekeeper, Band: tori_amos , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Winter, Album: Little_Earthquakes, Band: tori_amos , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 01-Pretty_Good_Year, Album: Under_the_Pink, Band: tori_amos , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 03-Sleeping_Single, Album: Look_Sharp_, Band: roxette , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-The_Centre_Of_The_Heart, Album: Room_Service, Band: roxette , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 12-My_World_My_Love_My_Life, Album: Room_Service, Band: roxette , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 11-Do_You_Wanna_Go_The_Whole_Way_, Album: Crash_Boom_Bang_, Band: roxette , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 08-Place_Your_Love, Album: Crash_Boom_Bang_, Band: roxette , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Heartbreaker, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 1.0000
    Requested track: track6.wav, Threshold:0.05 ,Song: 01-The_Song_Remains_The_Same, Album: Houses_of_the_Holy, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Out_On_The_Tiles, Album: Led_Zeppelin_III, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 08-I_Can_t_Quit_You_Baby, Album: Led_Zeppelin_I, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 02-Babe_I_m_Gonna_Leave_You, Album: Led_Zeppelin_I, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 03-Numb, Album: Zooropa, Band: u2 , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 01-Gloria, Album: October, Band: u2 , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Fire, Album: October, Band: u2 , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 08-With_a_Shout, Album: October, Band: u2 , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 01-Pagan_Baby, Album: Pendulum, Band: creedence_clearwater_revival , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 03-Travelin_Band, Album: Cosmo_s_Factory, Band: creedence_clearwater_revival , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.05 ,Song: 06-Lullaby, Album: Disintegration, Band: cure , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Last_Dance, Album: Disintegration, Band: cure , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 09-Bananafishbones, Album: The_Top, Band: cure , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 05-Dressing_Up, Album: The_Top, Band: cure , Similarity: 0.0667
    Requested track: track6.wav, Threshold:0.05 ,Song: 17-Fight, Album: Kiss_Me_Kiss_Me_Kiss_Me, Band: cure , Similarity: 0.0847
    Requested track: track6.wav, Threshold:0.1 ,Song: 07-Here_Comes_The_Sun, Album: Abbey_Road, Band: beatles , Similarity: 0.1228
    Requested track: track6.wav, Threshold:0.1 ,Song: 10-Louisiana_Bayou, Album: Stand_Up, Band: dave_matthews_band , Similarity: 0.1228
    Requested track: track6.wav, Threshold:0.1 ,Song: 10-Throw_Back_The_Little_Ones, Album: Katy_Lied, Band: steely_dan , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.1 ,Song: 02-My_Secret_Garden, Album: A_Broken_Frame, Band: depeche_mode , Similarity: 0.1228
    Requested track: track6.wav, Threshold:0.1 ,Song: 02-Widow_s_Walk, Album: Songs_in_Red_and_Gray, Band: suzanne_vega , Similarity: 0.1636
    Requested track: track6.wav, Threshold:0.1 ,Song: 13-The_Second_Time, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.1 ,Song: 03-Sleeping_Single, Album: Look_Sharp_, Band: roxette , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.1 ,Song: 08-Place_Your_Love, Album: Crash_Boom_Bang_, Band: roxette , Similarity: 0.1034
    Requested track: track6.wav, Threshold:0.1 ,Song: 05-Heartbreaker, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 1.0000
    Requested track: track6.wav, Threshold:0.25 ,Song: 05-Heartbreaker, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 1.0000
    Requested track: track6.wav, Threshold:0.3 ,Song: 05-Heartbreaker, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 1.0000
    Requested track: track7.wav, Threshold:0.05 ,Song: 07-She_Said_She_Said, Album: Revolver, Band: beatles , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 12-Piggies, Album: The_White_Album_Disc_1, Band: beatles , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 05-_41, Album: Crash, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 11-Pay_For_What_You_Get, Album: Under_The_Table_And_Dreaming, Band: dave_matthews_band , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 08-Everything_You_Did, Album: The_Royal_Scam, Band: steely_dan , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 02-Bad_Sneakers, Album: Katy_Lied, Band: steely_dan , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 08-Cure, Album: Load, Band: metallica , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 01-Ain_t_My_Bitch, Album: Load, Band: metallica , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 03-Adam_s_Apple, Album: Toys_In_The_Attic, Band: aerosmith , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 06-Sweet_Emotion, Album: Toys_In_The_Attic, Band: aerosmith , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 06-Train_Kept_A_Rollin_, Album: Get_Your_Wings, Band: aerosmith , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 02-Fly_on_the_windscreen, Album: Black_Celebration, Band: depeche_mode , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 13-To_have_and_to_hold_Spanish_Taster_, Album: Music_for_the_Masses, Band: depeche_mode , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 07-Master_And_Servant, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 01-Tired_Of_Sleeping, Album: Days_of_Open_Hand, Band: suzanne_vega , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 10-When_Heroes_Go_Down, Album: 99_9_F_, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 01-Penitent, Album: Songs_in_Red_and_Gray, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 09-Who_Needs_You, Album: News_Of_The_World, Band: queen , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 09-Love_of_My_Life, Album: A_Night_At_the_Opera, Band: queen , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 10-Misfire, Album: Sheer_Heart_Attack, Band: queen , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 02-Impressive_Instant, Album: Music, Band: madonna , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 13-Did_You_Do_It_, Album: Erotica, Band: madonna , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 17-Segue, Album: The_Love_Symbol_Album, Band: prince , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 03-Subterranean_Homesick_Alien, Album: OK_Computer, Band: radiohead , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 02-Love_Is_Dangerous, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 07-When_the_Sun_Goes_Down, Album: Behind_the_Mask, Band: fleetwood_mac , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 05-Go_Your_Own_Way, Album: Rumours, Band: fleetwood_mac , Similarity: 1.0000
    Requested track: track7.wav, Threshold:0.05 ,Song: 01-Crush_On_You, Album: Have_a_Nice_Day, Band: roxette , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 06-D_yer_Mak_er, Album: Houses_of_the_Holy, Band: led_zeppelin , Similarity: 0.0847
    Requested track: track7.wav, Threshold:0.05 ,Song: 07-In_God_s_Country, Album: The_Joshua_Tree, Band: u2 , Similarity: 0.1034
    Requested track: track7.wav, Threshold:0.05 ,Song: 01-Born_On_The_Bayou, Album: Bayou_Country, Band: creedence_clearwater_revival , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.05 ,Song: 04-If_Only_Tonight_We_Could_Sleep, Album: Kiss_Me_Kiss_Me_Kiss_Me, Band: cure , Similarity: 0.0667
    Requested track: track7.wav, Threshold:0.1 ,Song: 05-Go_Your_Own_Way, Album: Rumours, Band: fleetwood_mac , Similarity: 1.0000
    Requested track: track7.wav, Threshold:0.1 ,Song: 07-In_God_s_Country, Album: The_Joshua_Tree, Band: u2 , Similarity: 0.1034
    Requested track: track7.wav, Threshold:0.25 ,Song: 05-Go_Your_Own_Way, Album: Rumours, Band: fleetwood_mac , Similarity: 1.0000
    Requested track: track7.wav, Threshold:0.3 ,Song: 05-Go_Your_Own_Way, Album: Rumours, Band: fleetwood_mac , Similarity: 1.0000
    Requested track: track8.wav, Threshold:0.05 ,Song: 02-Fly_on_the_windscreen, Album: Black_Celebration, Band: depeche_mode , Similarity: 0.1034
    Requested track: track8.wav, Threshold:0.05 ,Song: 07-Master_And_Servant, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 0.0847
    Requested track: track8.wav, Threshold:0.05 ,Song: 03-Ironbound-Fancy_Poultry, Album: Solitude_Standing, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 11-Tom_s_Diner_Reprise, Album: Solitude_Standing, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 04-99_9_F_, Album: 99_9_F_, Band: suzanne_vega , Similarity: 0.1034
    Requested track: track8.wav, Threshold:0.05 ,Song: 01-American_Idiot, Album: American_Idiot, Band: green_day , Similarity: 1.0000
    Requested track: track8.wav, Threshold:0.05 ,Song: 05-Are_We_the_Waiting, Album: American_Idiot, Band: green_day , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 16-My_Generation, Album: Kerplunk, Band: green_day , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 09-Love_of_My_Life, Album: A_Night_At_the_Opera, Band: queen , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 05-Amazing, Album: Music, Band: madonna , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 08-More, Album: I_m_Breathless, Band: madonna , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 03-Subterranean_Homesick_Alien, Album: OK_Computer, Band: radiohead , Similarity: 0.1034
    Requested track: track8.wav, Threshold:0.05 ,Song: 13-Pearls_Of_Passion, Album: Pearls_of_Passion_1997_Remaster_, Band: roxette , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 03-The_Lemon_Song, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 07-Ramble_On, Album: Led_Zeppelin_II, Band: led_zeppelin , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 11-Mothers_of_the_Disappeared, Album: The_Joshua_Tree, Band: u2 , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 04-Have_You_Ever_Seen_The_Rain-, Album: Pendulum, Band: creedence_clearwater_revival , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 12-Trap, Album: Wild_Mood_Swings, Band: cure , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 06-Lullaby, Album: Disintegration, Band: cure , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 08-Prayers_For_Rain, Album: Disintegration, Band: cure , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.05 ,Song: 05-Push, Album: The_Head_on_the_Door, Band: cure , Similarity: 0.0847
    Requested track: track8.wav, Threshold:0.05 ,Song: 10-Cut, Album: Wish, Band: cure , Similarity: 0.0667
    Requested track: track8.wav, Threshold:0.1 ,Song: 02-Fly_on_the_windscreen, Album: Black_Celebration, Band: depeche_mode , Similarity: 0.1034
    Requested track: track8.wav, Threshold:0.1 ,Song: 04-99_9_F_, Album: 99_9_F_, Band: suzanne_vega , Similarity: 0.1034
    Requested track: track8.wav, Threshold:0.1 ,Song: 01-American_Idiot, Album: American_Idiot, Band: green_day , Similarity: 1.0000
    Requested track: track8.wav, Threshold:0.1 ,Song: 03-Subterranean_Homesick_Alien, Album: OK_Computer, Band: radiohead , Similarity: 0.1034
    Requested track: track8.wav, Threshold:0.25 ,Song: 01-American_Idiot, Album: American_Idiot, Band: green_day , Similarity: 1.0000
    Requested track: track8.wav, Threshold:0.3 ,Song: 01-American_Idiot, Album: American_Idiot, Band: green_day , Similarity: 1.0000
    Requested track: track9.wav, Threshold:0.05 ,Song: 06-Somebody, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 1.0000
    Requested track: track9.wav, Threshold:0.05 ,Song: 01-Music, Album: Music, Band: madonna , Similarity: 0.0847
    Requested track: track9.wav, Threshold:0.05 ,Song: 10-No_Surprises, Album: OK_Computer, Band: radiohead , Similarity: 0.0667
    Requested track: track9.wav, Threshold:0.1 ,Song: 06-Somebody, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 1.0000
    Requested track: track9.wav, Threshold:0.25 ,Song: 06-Somebody, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 1.0000
    Requested track: track9.wav, Threshold:0.3 ,Song: 06-Somebody, Album: Some_Great_Reward, Band: depeche_mode , Similarity: 1.0000
    Requested track: track10.wav, Threshold:0.05 ,Song: 06-Smooth_Rider, Album: Stand_Up, Band: dave_matthews_band , Similarity: 0.0847
    Requested track: track10.wav, Threshold:0.05 ,Song: 01-Black_Friday, Album: Katy_Lied, Band: steely_dan , Similarity: 1.0000
    Requested track: track10.wav, Threshold:0.05 ,Song: 08-King_Of_The_World, Album: Countdown_To_Ecstasy, Band: steely_dan , Similarity: 0.1034
    Requested track: track10.wav, Threshold:0.05 ,Song: 08-As_A_Child, Album: 99_9_F_, Band: suzanne_vega , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 11-Wake_Me_Up_When_September_Ends, Album: American_Idiot, Band: green_day , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 08-I_Wish_U_Heaven, Album: Lovesexy, Band: prince , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 04-Fake_Plastic_Trees, Album: The_Bends, Band: radiohead , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 14-A_Wolf_at_the_Door, Album: Hail_to_the_Theif, Band: radiohead , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 11-Wolves, Album: No_Fences_The_Limited_Series_, Band: garth_brooks , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 06-Dixie_Chicken, Album: The_Chase, Band: garth_brooks , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 02-Somewhere_Other_Than_The_Night, Album: The_Chase, Band: garth_brooks , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 11-Pay_The_Price, Album: Have_a_Nice_Day, Band: roxette , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 05-Bad_Moon_Rising, Album: Green_River, Band: creedence_clearwater_revival , Similarity: 0.0847
    Requested track: track10.wav, Threshold:0.05 ,Song: 03-Tombstone_Shadow, Album: Green_River, Band: creedence_clearwater_revival , Similarity: 0.0667
    Requested track: track10.wav, Threshold:0.05 ,Song: 05-_Wish_I_Could_Hideaway, Album: Pendulum, Band: creedence_clearwater_revival , Similarity: 0.0847
    Requested track: track10.wav, Threshold:0.1 ,Song: 01-Black_Friday, Album: Katy_Lied, Band: steely_dan , Similarity: 1.0000
    Requested track: track10.wav, Threshold:0.1 ,Song: 08-King_Of_The_World, Album: Countdown_To_Ecstasy, Band: steely_dan , Similarity: 0.1034
    Requested track: track10.wav, Threshold:0.25 ,Song: 01-Black_Friday, Album: Katy_Lied, Band: steely_dan , Similarity: 1.0000
    Requested track: track10.wav, Threshold:0.3 ,Song: 01-Black_Friday, Album: Katy_Lied, Band: steely_dan , Similarity: 1.0000
    

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------**

### Import utils and Dataframe


```python
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.spatial import distance
from sklearn.cluster import KMeans
```


```python
## Mount Drive
from google.colab import drive 
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
path = '/content/drive/My Drive/ADM/Homework IV'

import os
os.chdir(path)
```

## Question 2 - Grouping songs together

We play with a dataset gathering songs from the International Society for Music Information Retrieval Conference. The tracks (songs) include much information. We focus on the track information, features and audio variables. 

### 2.1 - Getting data

In this step we load the data.


```python
## Load Data
df1 = pd.read_csv('echonest.csv')
df2 = pd.read_csv('features.csv')
df3 = pd.read_csv('tracks.csv')
```

### Data Wrangling

After uploading data, you move to the wrangling data step, that is the process of gathering, selecting, and transforming data to answer an analytical question.

Starting from the first set of data, the main information such as number of observations, number of features and types of data are checked.


```python
## Info on df1
df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13129 entries, 0 to 13128
    Columns: 250 entries, track_id to temporal_features_223
    dtypes: float64(244), int64(1), object(5)
    memory usage: 25.0+ MB
    

The dataset contains 13129 observations for 250 features.

Then check the presence of null values within the dataset.


```python
## Check NaN values on first dataset
df1.isnull().sum()[0:50]
```




    track_id                                  0
    audio_features_acousticness               0
    audio_features_danceability               0
    audio_features_energy                     0
    audio_features_instrumentalness           0
    audio_features_liveness                   0
    audio_features_speechiness                0
    audio_features_tempo                      0
    audio_features_valence                    0
    metadata_album_date                   10412
    metadata_album_name                   10257
    metadata_artist_latitude               3359
    metadata_artist_location               3359
    metadata_artist_longitude              3359
    metadata_artist_name                      0
    metadata_release                          0
    ranks_artist_discovery_rank           10304
    ranks_artist_familiarity_rank         10304
    ranks_artist_hotttnesss_rank          10305
    ranks_song_currency_rank              11096
    ranks_song_hotttnesss_rank            10923
    social_features_artist_discovery          0
    social_features_artist_familiarity        0
    social_features_artist_hotttnesss         0
    social_features_song_currency             0
    social_features_song_hotttnesss           0
    temporal_features_000                     0
    temporal_features_001                     0
    temporal_features_002                     0
    temporal_features_003                     0
    temporal_features_004                     0
    temporal_features_005                     0
    temporal_features_006                     0
    temporal_features_007                     0
    temporal_features_008                     0
    temporal_features_009                     0
    temporal_features_010                     0
    temporal_features_011                     0
    temporal_features_012                     0
    temporal_features_013                     0
    temporal_features_014                     0
    temporal_features_015                     0
    temporal_features_016                     0
    temporal_features_017                     0
    temporal_features_018                     0
    temporal_features_019                     0
    temporal_features_020                     0
    temporal_features_021                     0
    temporal_features_022                     0
    temporal_features_023                     0
    dtype: int64



After a careful analysis of the null values for each variable, it emerges that some features such as **ranks_artist_discovery_rank** or **ranks_song_currency_rank**, for example, record more than 10000 null values, that is than 75% of the total observations. For this reason, an estimate of these values by interpolation would have been too distorted and therefore not precise and this leads us to the elimination of some variables listed below that contain more than 10000 null values.


```python
## Check the features with more and ugual than 10000 Null values (75% of observations).
df1.columns[df1.isnull().sum() >= 10000].tolist()
```




    ['metadata_album_date',
     'metadata_album_name',
     'ranks_artist_discovery_rank',
     'ranks_artist_familiarity_rank',
     'ranks_artist_hotttnesss_rank',
     'ranks_song_currency_rank',
     'ranks_song_hotttnesss_rank']




```python
## Drop the features with more than 10000 Null values
df1 = df1.drop(columns=['metadata_album_date', 'metadata_album_name', 'ranks_artist_discovery_rank', 'ranks_artist_familiarity_rank', 'ranks_artist_hotttnesss_rank',
                        'ranks_song_currency_rank', 'ranks_song_hotttnesss_rank'])
```

For feaures such as **metadata_artist_latitude**, **metadata_artist_location**, and **metadata_artist_longitude** the null values are 3359, so a lower percentage than the previous one, but still considerable. Again, the approach is to eliminate the features as in the previous case.


```python
## Drop the features 
df1 = df1.drop(columns=['metadata_artist_latitude','metadata_artist_location','metadata_artist_longitude'])
```


```python
## Check
df1.isnull().sum()[0:50]
```




    track_id                              0
    audio_features_acousticness           0
    audio_features_danceability           0
    audio_features_energy                 0
    audio_features_instrumentalness       0
    audio_features_liveness               0
    audio_features_speechiness            0
    audio_features_tempo                  0
    audio_features_valence                0
    metadata_artist_name                  0
    metadata_release                      0
    social_features_artist_discovery      0
    social_features_artist_familiarity    0
    social_features_artist_hotttnesss     0
    social_features_song_currency         0
    social_features_song_hotttnesss       0
    temporal_features_000                 0
    temporal_features_001                 0
    temporal_features_002                 0
    temporal_features_003                 0
    temporal_features_004                 0
    temporal_features_005                 0
    temporal_features_006                 0
    temporal_features_007                 0
    temporal_features_008                 0
    temporal_features_009                 0
    temporal_features_010                 0
    temporal_features_011                 0
    temporal_features_012                 0
    temporal_features_013                 0
    temporal_features_014                 0
    temporal_features_015                 0
    temporal_features_016                 0
    temporal_features_017                 0
    temporal_features_018                 0
    temporal_features_019                 0
    temporal_features_020                 0
    temporal_features_021                 0
    temporal_features_022                 0
    temporal_features_023                 0
    temporal_features_024                 0
    temporal_features_025                 0
    temporal_features_026                 0
    temporal_features_027                 0
    temporal_features_028                 0
    temporal_features_029                 0
    temporal_features_030                 0
    temporal_features_031                 0
    temporal_features_032                 0
    temporal_features_033                 0
    dtype: int64



The same operations carried out on the first dataset, are also repeated on the second.


```python
## Info on df2
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 106574 entries, 0 to 106573
    Columns: 519 entries, track_id to zcr_std_01
    dtypes: float64(518), int64(1)
    memory usage: 422.0 MB
    

The dataset contains 106574 observations for 519 features.


```python
## Check NaN values on second dataset
df2.isnull().sum()
```




    track_id                   0
    chroma_cens_kurtosis_01    0
    chroma_cens_kurtosis_02    0
    chroma_cens_kurtosis_03    0
    chroma_cens_kurtosis_04    0
                              ..
    zcr_mean_01                0
    zcr_median_01              0
    zcr_min_01                 0
    zcr_skew_01                0
    zcr_std_01                 0
    Length: 519, dtype: int64



In this case, there are no null values.

The same operations carried out on the third dataset.


```python
## Info on df3
df3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 106574 entries, 0 to 106573
    Data columns (total 53 columns):
     #   Column                    Non-Null Count   Dtype  
    ---  ------                    --------------   -----  
     0   track_id                  106574 non-null  int64  
     1   album_comments            106574 non-null  int64  
     2   album_date_created        103045 non-null  object 
     3   album_date_released       70294 non-null   object 
     4   album_engineer            15295 non-null   object 
     5   album_favorites           106574 non-null  int64  
     6   album_id                  106574 non-null  int64  
     7   album_information         83149 non-null   object 
     8   album_listens             106574 non-null  int64  
     9   album_producer            18060 non-null   object 
     10  album_tags                106574 non-null  object 
     11  album_title               105549 non-null  object 
     12  album_tracks              106574 non-null  int64  
     13  album_type                100066 non-null  object 
     14  artist_active_year_begin  22711 non-null   object 
     15  artist_active_year_end    5375 non-null    object 
     16  artist_associated_labels  14271 non-null   object 
     17  artist_bio                71156 non-null   object 
     18  artist_comments           106574 non-null  int64  
     19  artist_date_created       105718 non-null  object 
     20  artist_favorites          106574 non-null  int64  
     21  artist_id                 106574 non-null  int64  
     22  artist_latitude           44544 non-null   float64
     23  artist_location           70210 non-null   object 
     24  artist_longitude          44544 non-null   float64
     25  artist_members            46849 non-null   object 
     26  artist_name               106574 non-null  object 
     27  artist_related_projects   13152 non-null   object 
     28  artist_tags               106574 non-null  object 
     29  artist_website            79256 non-null   object 
     30  artist_wikipedia_page     5581 non-null    object 
     31  set_split                 106574 non-null  object 
     32  set_subset                106574 non-null  object 
     33  track_bit_rate            106574 non-null  int64  
     34  track_comments            106574 non-null  int64  
     35  track_composer            3670 non-null    object 
     36  track_date_created        106574 non-null  object 
     37  track_date_recorded       6159 non-null    object 
     38  track_duration            106574 non-null  int64  
     39  track_favorites           106574 non-null  int64  
     40  track_genre_top           49598 non-null   object 
     41  track_genres              106574 non-null  object 
     42  track_genres_all          106574 non-null  object 
     43  track_information         2349 non-null    object 
     44  track_interest            106574 non-null  int64  
     45  track_language_code       15024 non-null   object 
     46  track_license             106487 non-null  object 
     47  track_listens             106574 non-null  int64  
     48  track_lyricist            311 non-null     object 
     49  track_number              106574 non-null  int64  
     50  track_publisher           1263 non-null    object 
     51  track_tags                106574 non-null  object 
     52  track_title               106573 non-null  object 
    dtypes: float64(2), int64(16), object(35)
    memory usage: 43.1+ MB
    

The dataset contains 106574 observations for 53 features.


```python
## Check NaN values on second dataset
df3.isnull().sum()
```




    track_id                         0
    album_comments                   0
    album_date_created            3529
    album_date_released          36280
    album_engineer               91279
    album_favorites                  0
    album_id                         0
    album_information            23425
    album_listens                    0
    album_producer               88514
    album_tags                       0
    album_title                   1025
    album_tracks                     0
    album_type                    6508
    artist_active_year_begin     83863
    artist_active_year_end      101199
    artist_associated_labels     92303
    artist_bio                   35418
    artist_comments                  0
    artist_date_created            856
    artist_favorites                 0
    artist_id                        0
    artist_latitude              62030
    artist_location              36364
    artist_longitude             62030
    artist_members               59725
    artist_name                      0
    artist_related_projects      93422
    artist_tags                      0
    artist_website               27318
    artist_wikipedia_page       100993
    set_split                        0
    set_subset                       0
    track_bit_rate                   0
    track_comments                   0
    track_composer              102904
    track_date_created               0
    track_date_recorded         100415
    track_duration                   0
    track_favorites                  0
    track_genre_top              56976
    track_genres                     0
    track_genres_all                 0
    track_information           104225
    track_interest                   0
    track_language_code          91550
    track_license                   87
    track_listens                    0
    track_lyricist              106263
    track_number                     0
    track_publisher             105311
    track_tags                       0
    track_title                      1
    dtype: int64



In the third dataset, there are many features that have null values. However, the interest in this case shifts to the variables that may serve to make the clustering of songs. For this reason, features such as **song’s duration**, **language**, **country** and **genre**.

Extract the fratures of interest by third dataset: song's duration, language, country and genre.


```python
## Song's duration
song_duration = df3.track_duration
```


```python
## Language
song_language = df3.track_language_code
```


```python
## Genre
song_genre = df3.track_genre_top 
```

At this point, you join the first and second datasets using the common key **track_id**. Given the type of features present in the third dataset, we choose to keep from this only the features listed above and then add them to the dataset transformed with the PCA to keep the original interpretation.


```python
## Interscetion df1 Vs. df2
set(df1.columns).intersection(set(df2.columns))
```




    {'track_id'}




```python
## Join
df = pd.merge(left = df1, right = df2, left_on = 'track_id', right_on = 'track_id')
```


```python
## Info on data 
df.info()
## Check the rows of dataset(13K)...it's right.
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 13129 entries, 0 to 13128
    Columns: 758 entries, track_id to zcr_std_01
    dtypes: float64(755), int64(1), object(2)
    memory usage: 76.0+ MB
    

The final dataset contains 13129 observations and 761 features, obviously without missing values.


```python
## Check
df.columns[df.isnull().sum() >= 1].tolist()
```




    []



The final dataset looks like this...


```python
## Check the head of dataset.
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>audio_features_acousticness</th>
      <th>audio_features_danceability</th>
      <th>audio_features_energy</th>
      <th>audio_features_instrumentalness</th>
      <th>audio_features_liveness</th>
      <th>audio_features_speechiness</th>
      <th>audio_features_tempo</th>
      <th>audio_features_valence</th>
      <th>metadata_artist_name</th>
      <th>metadata_release</th>
      <th>social_features_artist_discovery</th>
      <th>social_features_artist_familiarity</th>
      <th>social_features_artist_hotttnesss</th>
      <th>social_features_song_currency</th>
      <th>social_features_song_hotttnesss</th>
      <th>temporal_features_000</th>
      <th>temporal_features_001</th>
      <th>temporal_features_002</th>
      <th>temporal_features_003</th>
      <th>temporal_features_004</th>
      <th>temporal_features_005</th>
      <th>temporal_features_006</th>
      <th>temporal_features_007</th>
      <th>temporal_features_008</th>
      <th>temporal_features_009</th>
      <th>temporal_features_010</th>
      <th>temporal_features_011</th>
      <th>temporal_features_012</th>
      <th>temporal_features_013</th>
      <th>temporal_features_014</th>
      <th>temporal_features_015</th>
      <th>temporal_features_016</th>
      <th>temporal_features_017</th>
      <th>temporal_features_018</th>
      <th>temporal_features_019</th>
      <th>temporal_features_020</th>
      <th>temporal_features_021</th>
      <th>temporal_features_022</th>
      <th>temporal_features_023</th>
      <th>...</th>
      <th>tonnetz_max_04</th>
      <th>tonnetz_max_05</th>
      <th>tonnetz_max_06</th>
      <th>tonnetz_mean_01</th>
      <th>tonnetz_mean_02</th>
      <th>tonnetz_mean_03</th>
      <th>tonnetz_mean_04</th>
      <th>tonnetz_mean_05</th>
      <th>tonnetz_mean_06</th>
      <th>tonnetz_median_01</th>
      <th>tonnetz_median_02</th>
      <th>tonnetz_median_03</th>
      <th>tonnetz_median_04</th>
      <th>tonnetz_median_05</th>
      <th>tonnetz_median_06</th>
      <th>tonnetz_min_01</th>
      <th>tonnetz_min_02</th>
      <th>tonnetz_min_03</th>
      <th>tonnetz_min_04</th>
      <th>tonnetz_min_05</th>
      <th>tonnetz_min_06</th>
      <th>tonnetz_skew_01</th>
      <th>tonnetz_skew_02</th>
      <th>tonnetz_skew_03</th>
      <th>tonnetz_skew_04</th>
      <th>tonnetz_skew_05</th>
      <th>tonnetz_skew_06</th>
      <th>tonnetz_std_01</th>
      <th>tonnetz_std_02</th>
      <th>tonnetz_std_03</th>
      <th>tonnetz_std_04</th>
      <th>tonnetz_std_05</th>
      <th>tonnetz_std_06</th>
      <th>zcr_kurtosis_01</th>
      <th>zcr_max_01</th>
      <th>zcr_mean_01</th>
      <th>zcr_median_01</th>
      <th>zcr_min_01</th>
      <th>zcr_skew_01</th>
      <th>zcr_std_01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.416675</td>
      <td>0.675894</td>
      <td>0.634476</td>
      <td>0.010628</td>
      <td>0.177647</td>
      <td>0.159310</td>
      <td>165.922</td>
      <td>0.576661</td>
      <td>AWOL</td>
      <td>AWOL - A Way Of Life</td>
      <td>0.388990</td>
      <td>0.386740</td>
      <td>0.406370</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.877233</td>
      <td>0.588911</td>
      <td>0.354243</td>
      <td>0.295090</td>
      <td>0.298413</td>
      <td>0.309430</td>
      <td>0.304496</td>
      <td>0.334579</td>
      <td>0.249495</td>
      <td>0.259656</td>
      <td>0.318376</td>
      <td>0.371974</td>
      <td>1.000</td>
      <td>0.5710</td>
      <td>0.278</td>
      <td>0.2100</td>
      <td>0.2150</td>
      <td>0.2285</td>
      <td>0.2375</td>
      <td>0.279</td>
      <td>0.1685</td>
      <td>0.1685</td>
      <td>0.279</td>
      <td>0.3325</td>
      <td>...</td>
      <td>0.318972</td>
      <td>0.059690</td>
      <td>0.069184</td>
      <td>-0.002570</td>
      <td>0.019296</td>
      <td>0.010510</td>
      <td>0.073464</td>
      <td>0.009272</td>
      <td>0.015765</td>
      <td>-0.003789</td>
      <td>0.017786</td>
      <td>0.007311</td>
      <td>0.067945</td>
      <td>0.009488</td>
      <td>0.016876</td>
      <td>-0.059769</td>
      <td>-0.091745</td>
      <td>-0.185687</td>
      <td>-0.140306</td>
      <td>-0.048525</td>
      <td>-0.089286</td>
      <td>0.752462</td>
      <td>0.262607</td>
      <td>0.200944</td>
      <td>0.593595</td>
      <td>-0.177665</td>
      <td>-1.424201</td>
      <td>0.019809</td>
      <td>0.029569</td>
      <td>0.038974</td>
      <td>0.054125</td>
      <td>0.012226</td>
      <td>0.012111</td>
      <td>5.758890</td>
      <td>0.459473</td>
      <td>0.085629</td>
      <td>0.071289</td>
      <td>0.0</td>
      <td>2.089872</td>
      <td>0.061448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.374408</td>
      <td>0.528643</td>
      <td>0.817461</td>
      <td>0.001851</td>
      <td>0.105880</td>
      <td>0.461818</td>
      <td>126.957</td>
      <td>0.269240</td>
      <td>AWOL</td>
      <td>AWOL - A Way Of Life</td>
      <td>0.388990</td>
      <td>0.386740</td>
      <td>0.406370</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.534429</td>
      <td>0.537414</td>
      <td>0.443299</td>
      <td>0.390879</td>
      <td>0.344573</td>
      <td>0.366448</td>
      <td>0.419455</td>
      <td>0.747766</td>
      <td>0.460901</td>
      <td>0.392379</td>
      <td>0.474559</td>
      <td>0.406729</td>
      <td>0.506</td>
      <td>0.5145</td>
      <td>0.387</td>
      <td>0.3235</td>
      <td>0.2805</td>
      <td>0.3135</td>
      <td>0.3455</td>
      <td>0.898</td>
      <td>0.4365</td>
      <td>0.3385</td>
      <td>0.398</td>
      <td>0.3480</td>
      <td>...</td>
      <td>0.214807</td>
      <td>0.070261</td>
      <td>0.070394</td>
      <td>0.000183</td>
      <td>0.006908</td>
      <td>0.047025</td>
      <td>-0.029942</td>
      <td>0.017535</td>
      <td>-0.001496</td>
      <td>-0.000108</td>
      <td>0.007161</td>
      <td>0.046912</td>
      <td>-0.021149</td>
      <td>0.016299</td>
      <td>-0.002657</td>
      <td>-0.097199</td>
      <td>-0.079651</td>
      <td>-0.164613</td>
      <td>-0.304375</td>
      <td>-0.024958</td>
      <td>-0.055667</td>
      <td>0.265541</td>
      <td>-0.131471</td>
      <td>0.171930</td>
      <td>-0.990710</td>
      <td>0.574556</td>
      <td>0.556494</td>
      <td>0.026316</td>
      <td>0.018708</td>
      <td>0.051151</td>
      <td>0.063831</td>
      <td>0.014212</td>
      <td>0.017740</td>
      <td>2.824694</td>
      <td>0.466309</td>
      <td>0.084578</td>
      <td>0.063965</td>
      <td>0.0</td>
      <td>1.716724</td>
      <td>0.069330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.043567</td>
      <td>0.745566</td>
      <td>0.701470</td>
      <td>0.000697</td>
      <td>0.373143</td>
      <td>0.124595</td>
      <td>100.260</td>
      <td>0.621661</td>
      <td>AWOL</td>
      <td>AWOL - A Way Of Life</td>
      <td>0.388990</td>
      <td>0.386740</td>
      <td>0.406370</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.548093</td>
      <td>0.720192</td>
      <td>0.389257</td>
      <td>0.344934</td>
      <td>0.361300</td>
      <td>0.402543</td>
      <td>0.434044</td>
      <td>0.388137</td>
      <td>0.512487</td>
      <td>0.525755</td>
      <td>0.425371</td>
      <td>0.446896</td>
      <td>0.511</td>
      <td>0.7720</td>
      <td>0.361</td>
      <td>0.2880</td>
      <td>0.3310</td>
      <td>0.3720</td>
      <td>0.3590</td>
      <td>0.279</td>
      <td>0.4430</td>
      <td>0.4840</td>
      <td>0.368</td>
      <td>0.3970</td>
      <td>...</td>
      <td>0.180027</td>
      <td>0.072169</td>
      <td>0.076847</td>
      <td>-0.007501</td>
      <td>-0.018525</td>
      <td>-0.030318</td>
      <td>0.024743</td>
      <td>0.004771</td>
      <td>-0.004536</td>
      <td>-0.007385</td>
      <td>-0.018953</td>
      <td>-0.020358</td>
      <td>0.024615</td>
      <td>0.004868</td>
      <td>-0.003899</td>
      <td>-0.128391</td>
      <td>-0.125289</td>
      <td>-0.359463</td>
      <td>-0.166667</td>
      <td>-0.038546</td>
      <td>-0.146136</td>
      <td>1.212025</td>
      <td>0.218381</td>
      <td>-0.419971</td>
      <td>-0.014541</td>
      <td>-0.199314</td>
      <td>-0.925733</td>
      <td>0.025550</td>
      <td>0.021106</td>
      <td>0.084997</td>
      <td>0.040730</td>
      <td>0.012691</td>
      <td>0.014759</td>
      <td>6.808415</td>
      <td>0.375000</td>
      <td>0.053114</td>
      <td>0.041504</td>
      <td>0.0</td>
      <td>2.193303</td>
      <td>0.044861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>0.951670</td>
      <td>0.658179</td>
      <td>0.924525</td>
      <td>0.965427</td>
      <td>0.115474</td>
      <td>0.032985</td>
      <td>111.562</td>
      <td>0.963590</td>
      <td>Kurt Vile</td>
      <td>Constant Hitmaker</td>
      <td>0.557339</td>
      <td>0.614272</td>
      <td>0.798387</td>
      <td>0.005158</td>
      <td>0.354516</td>
      <td>0.311404</td>
      <td>0.711402</td>
      <td>0.321914</td>
      <td>0.500601</td>
      <td>0.250963</td>
      <td>0.321316</td>
      <td>0.734250</td>
      <td>0.325188</td>
      <td>0.373012</td>
      <td>0.235840</td>
      <td>0.368756</td>
      <td>0.440775</td>
      <td>0.263</td>
      <td>0.7360</td>
      <td>0.273</td>
      <td>0.4260</td>
      <td>0.2140</td>
      <td>0.2880</td>
      <td>0.8100</td>
      <td>0.246</td>
      <td>0.2950</td>
      <td>0.1640</td>
      <td>0.311</td>
      <td>0.3860</td>
      <td>...</td>
      <td>0.192640</td>
      <td>0.117094</td>
      <td>0.059757</td>
      <td>-0.021650</td>
      <td>-0.018369</td>
      <td>-0.003282</td>
      <td>-0.074165</td>
      <td>0.008971</td>
      <td>0.007101</td>
      <td>-0.021108</td>
      <td>-0.019117</td>
      <td>-0.007409</td>
      <td>-0.067350</td>
      <td>0.007036</td>
      <td>0.006788</td>
      <td>-0.107889</td>
      <td>-0.194957</td>
      <td>-0.273549</td>
      <td>-0.343055</td>
      <td>-0.052284</td>
      <td>-0.029836</td>
      <td>-0.135219</td>
      <td>-0.275780</td>
      <td>0.015767</td>
      <td>-1.094873</td>
      <td>1.164041</td>
      <td>0.246746</td>
      <td>0.021413</td>
      <td>0.031989</td>
      <td>0.088197</td>
      <td>0.074358</td>
      <td>0.017952</td>
      <td>0.013921</td>
      <td>21.434212</td>
      <td>0.452148</td>
      <td>0.077515</td>
      <td>0.071777</td>
      <td>0.0</td>
      <td>3.542325</td>
      <td>0.040800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>134</td>
      <td>0.452217</td>
      <td>0.513238</td>
      <td>0.560410</td>
      <td>0.019443</td>
      <td>0.096567</td>
      <td>0.525519</td>
      <td>114.290</td>
      <td>0.894072</td>
      <td>AWOL</td>
      <td>AWOL - A Way Of Life</td>
      <td>0.388990</td>
      <td>0.386740</td>
      <td>0.406370</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.610849</td>
      <td>0.569169</td>
      <td>0.428494</td>
      <td>0.345796</td>
      <td>0.376920</td>
      <td>0.460590</td>
      <td>0.401371</td>
      <td>0.449900</td>
      <td>0.428946</td>
      <td>0.446736</td>
      <td>0.479849</td>
      <td>0.378221</td>
      <td>0.614</td>
      <td>0.5450</td>
      <td>0.363</td>
      <td>0.2800</td>
      <td>0.3110</td>
      <td>0.3970</td>
      <td>0.3170</td>
      <td>0.404</td>
      <td>0.3560</td>
      <td>0.3800</td>
      <td>0.420</td>
      <td>0.2920</td>
      <td>...</td>
      <td>0.236018</td>
      <td>0.076754</td>
      <td>0.090965</td>
      <td>0.007164</td>
      <td>0.021010</td>
      <td>-0.045582</td>
      <td>-0.021565</td>
      <td>0.007772</td>
      <td>0.000070</td>
      <td>0.004903</td>
      <td>0.020440</td>
      <td>-0.047713</td>
      <td>-0.014039</td>
      <td>0.006792</td>
      <td>0.000000</td>
      <td>-0.063484</td>
      <td>-0.225944</td>
      <td>-0.230290</td>
      <td>-0.328776</td>
      <td>-0.049787</td>
      <td>-0.053569</td>
      <td>0.927807</td>
      <td>-0.947771</td>
      <td>0.143864</td>
      <td>-0.529867</td>
      <td>0.162188</td>
      <td>0.063846</td>
      <td>0.024258</td>
      <td>0.028818</td>
      <td>0.060806</td>
      <td>0.058766</td>
      <td>0.016322</td>
      <td>0.015819</td>
      <td>4.731087</td>
      <td>0.419434</td>
      <td>0.064370</td>
      <td>0.050781</td>
      <td>0.0</td>
      <td>1.806106</td>
      <td>0.054623</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 758 columns</p>
</div>



A final cleanup of the data before proceeding with the next step concerns the removal of the variable **track_id** because it represents a unique code of the song and therefore it makes no sense to include it within the PCA and the removal of some categorical features that cannot be included within the PCA and K-means at a later date.


```python
## Drop columns
df = df.drop(columns=['track_id', 'metadata_artist_name', 'metadata_release'])
```


```python
## Info
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 13129 entries, 0 to 13128
    Columns: 755 entries, audio_features_acousticness to zcr_std_01
    dtypes: float64(755)
    memory usage: 75.7 MB
    

Now you can apply a method for the reduction of dimensionality.

### 2.2 - Choose your features

Before defining the method used in this case, it is necessary to define what is generally meant by reduction of dimensionality.

### 2.2.1 Problem of the reduction

I observe a vector of $x_i \in R^{p}$, I look for a way to map the data
$x_i \xrightarrow{} z_i \in R^{q}$ ; with $q < p$
making sure that:
- Preserve some relevant information of the distribution of $X$.
- I can return to the starting space $z_i \xrightarrow{} x_i$.

Example: I want to map red dots $x_i \in R^{2}$ to $R$ so I don’t
lose information about centrality/dispersion.
![Screenshot (63).png](data:image/png;base64,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)

Returning to the example of before: wanting to memorize all the red dots I need to memorize the $2 \times n$ values $f({xi_1; xi_2})$,
exploiting the fact that the points are on the line:

- $X_2 = a + bX_1$

I could, for example, only memorize the first coordinate
$(x1_1; x2_1; ... ; xn_1)$ and then the coefficient pair $(a; b)$, then pass
from $2n$ values to $n + 2$ values.

### 2.2.2 - Principal Component Analysis (PCA)

The **PCA** technique was chosen for the reduction of dimensionality.

**Advantages**:

- Removes Correlated Features
- Improves Algorithm Performance
- Reduces Overfitting
-  Improves Visualization

**Goal**:
- transform the original variables $X = (X_1, X_2, ..., X_p)'$ in $Z = (Z_1, Z_2, ..., Z_p)'$. 
- The transformation is linear type: $Z_j = ϕ_{1_{j}} \times X_1 + ϕ_{2_{j}} \times X_2 + ... + ϕ_{p_{j}} \times X_p.$
- The coefficients $ϕ_j$ are called **loadings** of the components.
- The main components are ordered by decreasing and incorrectly distributed variance.
We also want the Zs to preserve all the variance content in X so as to represent just another dimension in which to look at the data, but without losing information. Only later, with the analysis of the main components, it is decided how much part of variability, where by variability is meant information, capture with the components.
There is no defined number of variance to cumulate, it depends a lot on the specific case. Usually it does not fall below 70% and in some cases even 80%, such as in the compression of pixels of images.

Before proceeding with the PCA it is necessary to standardize the features.
This step is important, as the PCA searches for components along the directions in which variance is maximized. If we did the exercise of plotting the original variances for each variable, these would be misleading as they would suggest that only variables with greater variance contribute to the information of the data, while standardizing the variables, The situation changes and improves PCA performance.


```python
## Standardization
scaler = StandardScaler()
df_scale = scaler.fit_transform(df)
df_scale = pd.DataFrame(df_scale)
df_scale.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>...</th>
      <th>715</th>
      <th>716</th>
      <th>717</th>
      <th>718</th>
      <th>719</th>
      <th>720</th>
      <th>721</th>
      <th>722</th>
      <th>723</th>
      <th>724</th>
      <th>725</th>
      <th>726</th>
      <th>727</th>
      <th>728</th>
      <th>729</th>
      <th>730</th>
      <th>731</th>
      <th>732</th>
      <th>733</th>
      <th>734</th>
      <th>735</th>
      <th>736</th>
      <th>737</th>
      <th>738</th>
      <th>739</th>
      <th>740</th>
      <th>741</th>
      <th>742</th>
      <th>743</th>
      <th>744</th>
      <th>745</th>
      <th>746</th>
      <th>747</th>
      <th>748</th>
      <th>749</th>
      <th>750</th>
      <th>751</th>
      <th>752</th>
      <th>753</th>
      <th>754</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.281499</td>
      <td>0.991915</td>
      <td>0.348732</td>
      <td>-1.742887</td>
      <td>-0.064267</td>
      <td>0.437749</td>
      <td>1.223573</td>
      <td>0.495983</td>
      <td>0.584512</td>
      <td>0.782963</td>
      <td>0.447205</td>
      <td>-0.15861</td>
      <td>-0.537540</td>
      <td>2.602372</td>
      <td>0.919566</td>
      <td>-0.082752</td>
      <td>-0.092399</td>
      <td>-0.481088</td>
      <td>-0.130599</td>
      <td>-0.224785</td>
      <td>-0.178109</td>
      <td>-0.586609</td>
      <td>-0.732929</td>
      <td>0.149177</td>
      <td>0.245649</td>
      <td>2.655370</td>
      <td>0.828443</td>
      <td>-0.099867</td>
      <td>-0.239930</td>
      <td>-0.471167</td>
      <td>-0.179055</td>
      <td>-0.174634</td>
      <td>-0.015079</td>
      <td>-0.590848</td>
      <td>-0.664598</td>
      <td>0.401624</td>
      <td>0.435481</td>
      <td>-1.187552</td>
      <td>0.768230</td>
      <td>-0.514756</td>
      <td>...</td>
      <td>-0.024206</td>
      <td>-0.695639</td>
      <td>-0.184375</td>
      <td>-0.281040</td>
      <td>1.230567</td>
      <td>0.180082</td>
      <td>1.029232</td>
      <td>0.837030</td>
      <td>1.520077</td>
      <td>-0.414588</td>
      <td>1.149987</td>
      <td>0.121949</td>
      <td>0.884728</td>
      <td>0.859322</td>
      <td>1.576897</td>
      <td>1.324580</td>
      <td>0.507170</td>
      <td>1.322830</td>
      <td>1.691190</td>
      <td>0.994173</td>
      <td>-0.602813</td>
      <td>1.493278</td>
      <td>0.343301</td>
      <td>0.323768</td>
      <td>1.185503</td>
      <td>-0.526961</td>
      <td>-3.213633</td>
      <td>-1.376319</td>
      <td>-0.040156</td>
      <td>-1.737526</td>
      <td>-1.310092</td>
      <td>-1.564443</td>
      <td>-1.654377</td>
      <td>-0.381721</td>
      <td>0.393540</td>
      <td>1.208676</td>
      <td>1.035766</td>
      <td>-0.594301</td>
      <td>-0.378710</td>
      <td>0.932033</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.391656</td>
      <td>0.217484</td>
      <td>1.006860</td>
      <td>-1.767172</td>
      <td>-0.518358</td>
      <td>2.639805</td>
      <td>0.110726</td>
      <td>-0.617788</td>
      <td>0.584512</td>
      <td>0.782963</td>
      <td>0.447205</td>
      <td>-0.15861</td>
      <td>-0.537540</td>
      <td>0.522280</td>
      <td>0.610138</td>
      <td>0.589480</td>
      <td>0.713942</td>
      <td>-0.150159</td>
      <td>0.326192</td>
      <td>0.701000</td>
      <td>2.969636</td>
      <td>1.126624</td>
      <td>0.253437</td>
      <td>1.499448</td>
      <td>0.513654</td>
      <td>0.463686</td>
      <td>0.572093</td>
      <td>0.556746</td>
      <td>0.617020</td>
      <td>-0.094269</td>
      <td>0.412044</td>
      <td>0.581684</td>
      <td>3.813134</td>
      <td>1.321576</td>
      <td>0.346295</td>
      <td>1.331264</td>
      <td>0.534571</td>
      <td>-0.219016</td>
      <td>0.051239</td>
      <td>-0.088406</td>
      <td>...</td>
      <td>-1.149175</td>
      <td>-0.239190</td>
      <td>-0.130131</td>
      <td>-0.041981</td>
      <td>0.289546</td>
      <td>0.810686</td>
      <td>-0.695398</td>
      <td>1.772551</td>
      <td>-0.334457</td>
      <td>-0.087693</td>
      <td>0.317707</td>
      <td>0.766449</td>
      <td>-0.529523</td>
      <td>1.615245</td>
      <td>-0.494903</td>
      <td>0.475378</td>
      <td>0.790525</td>
      <td>1.548409</td>
      <td>-0.001003</td>
      <td>2.089565</td>
      <td>0.841209</td>
      <td>0.620469</td>
      <td>-0.287215</td>
      <td>0.267141</td>
      <td>-1.841999</td>
      <td>1.400393</td>
      <td>1.632515</td>
      <td>-0.426613</td>
      <td>-1.525809</td>
      <td>-1.381420</td>
      <td>-1.021250</td>
      <td>-1.202410</td>
      <td>-0.608422</td>
      <td>-0.439876</td>
      <td>0.432798</td>
      <td>1.168234</td>
      <td>0.742018</td>
      <td>-0.594301</td>
      <td>-0.526657</td>
      <td>1.267315</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.253885</td>
      <td>1.358338</td>
      <td>0.589683</td>
      <td>-1.770366</td>
      <td>1.172705</td>
      <td>0.185049</td>
      <td>-0.651745</td>
      <td>0.659017</td>
      <td>0.584512</td>
      <td>0.782963</td>
      <td>0.447205</td>
      <td>-0.15861</td>
      <td>-0.537540</td>
      <td>0.605188</td>
      <td>1.708388</td>
      <td>0.181548</td>
      <td>0.327181</td>
      <td>-0.030244</td>
      <td>0.615368</td>
      <td>0.818482</td>
      <td>0.229910</td>
      <td>1.544675</td>
      <td>1.244655</td>
      <td>1.074195</td>
      <td>0.823395</td>
      <td>0.485869</td>
      <td>1.740413</td>
      <td>0.400123</td>
      <td>0.348987</td>
      <td>0.196316</td>
      <td>0.818859</td>
      <td>0.676223</td>
      <td>-0.015079</td>
      <td>1.367959</td>
      <td>1.211500</td>
      <td>1.096901</td>
      <td>0.847824</td>
      <td>-0.158188</td>
      <td>-0.120999</td>
      <td>-0.882645</td>
      <td>...</td>
      <td>-1.524798</td>
      <td>-0.156829</td>
      <td>0.159060</td>
      <td>-0.709186</td>
      <td>-1.642357</td>
      <td>-0.524986</td>
      <td>0.216642</td>
      <td>0.327335</td>
      <td>-0.661011</td>
      <td>-0.733933</td>
      <td>-1.727929</td>
      <td>-0.328367</td>
      <td>0.196931</td>
      <td>0.346507</td>
      <td>-0.626618</td>
      <td>-0.232317</td>
      <td>-0.278688</td>
      <td>-0.537342</td>
      <td>1.419312</td>
      <td>1.457991</td>
      <td>-3.044718</td>
      <td>2.317048</td>
      <td>0.272540</td>
      <td>-0.888083</td>
      <td>0.023395</td>
      <td>-0.582432</td>
      <td>-1.994035</td>
      <td>-0.538400</td>
      <td>-1.197730</td>
      <td>-0.391558</td>
      <td>-1.708717</td>
      <td>-1.479675</td>
      <td>-1.162289</td>
      <td>-0.360920</td>
      <td>-0.091584</td>
      <td>-0.042573</td>
      <td>-0.158812</td>
      <td>-0.594301</td>
      <td>-0.337701</td>
      <td>0.226443</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.112791</td>
      <td>0.898745</td>
      <td>1.391929</td>
      <td>0.898939</td>
      <td>-0.457654</td>
      <td>-0.481812</td>
      <td>-0.328958</td>
      <td>1.897809</td>
      <td>2.053342</td>
      <td>2.556413</td>
      <td>3.388650</td>
      <td>3.34078</td>
      <td>5.183888</td>
      <td>-0.831008</td>
      <td>1.655575</td>
      <td>-0.326785</td>
      <td>1.637571</td>
      <td>-0.821258</td>
      <td>-0.035374</td>
      <td>3.236081</td>
      <td>-0.249650</td>
      <td>0.414377</td>
      <td>-0.909919</td>
      <td>0.584729</td>
      <td>0.776190</td>
      <td>-0.614409</td>
      <td>1.577075</td>
      <td>-0.129987</td>
      <td>1.390918</td>
      <td>-0.476921</td>
      <td>0.234714</td>
      <td>3.834551</td>
      <td>-0.219168</td>
      <td>0.311844</td>
      <td>-0.691357</td>
      <td>0.651611</td>
      <td>0.777502</td>
      <td>-1.711390</td>
      <td>-0.382097</td>
      <td>-1.186813</td>
      <td>...</td>
      <td>-1.388580</td>
      <td>1.782949</td>
      <td>-0.606816</td>
      <td>-1.937858</td>
      <td>-1.630512</td>
      <td>-0.058095</td>
      <td>-1.432959</td>
      <td>0.802858</td>
      <td>0.589182</td>
      <td>-1.952502</td>
      <td>-1.740793</td>
      <td>-0.117614</td>
      <td>-1.262910</td>
      <td>0.587165</td>
      <td>0.506931</td>
      <td>0.232828</td>
      <td>-1.910872</td>
      <td>0.382314</td>
      <td>-0.399938</td>
      <td>0.819464</td>
      <td>1.950749</td>
      <td>-0.097895</td>
      <td>-0.518105</td>
      <td>-0.037645</td>
      <td>-2.041049</td>
      <td>2.910786</td>
      <td>0.874657</td>
      <td>-1.142293</td>
      <td>0.290955</td>
      <td>-0.297983</td>
      <td>-0.707981</td>
      <td>-0.520648</td>
      <td>-1.317931</td>
      <td>-0.071044</td>
      <td>0.351477</td>
      <td>0.896399</td>
      <td>1.055350</td>
      <td>-0.594301</td>
      <td>0.197164</td>
      <td>0.053735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.188870</td>
      <td>0.136465</td>
      <td>0.082343</td>
      <td>-1.718498</td>
      <td>-0.577286</td>
      <td>3.103507</td>
      <td>-0.251046</td>
      <td>1.645950</td>
      <td>0.584512</td>
      <td>0.782963</td>
      <td>0.447205</td>
      <td>-0.15861</td>
      <td>-0.537540</td>
      <td>0.985987</td>
      <td>0.800945</td>
      <td>0.477721</td>
      <td>0.334437</td>
      <td>0.081742</td>
      <td>1.080411</td>
      <td>0.555364</td>
      <td>0.700433</td>
      <td>0.867666</td>
      <td>0.657402</td>
      <td>1.545186</td>
      <td>0.293827</td>
      <td>0.942840</td>
      <td>0.710477</td>
      <td>0.412171</td>
      <td>0.288586</td>
      <td>0.081233</td>
      <td>0.992712</td>
      <td>0.382100</td>
      <td>0.757985</td>
      <td>0.747135</td>
      <td>0.593071</td>
      <td>1.503130</td>
      <td>0.176569</td>
      <td>-0.022105</td>
      <td>0.357474</td>
      <td>-0.101561</td>
      <td>...</td>
      <td>-0.920102</td>
      <td>0.041167</td>
      <td>0.791719</td>
      <td>0.564267</td>
      <td>1.360788</td>
      <td>-0.788587</td>
      <td>-0.555680</td>
      <td>0.667138</td>
      <td>-0.166224</td>
      <td>0.357281</td>
      <td>1.357913</td>
      <td>-0.773565</td>
      <td>-0.416653</td>
      <td>0.560142</td>
      <td>-0.213075</td>
      <td>1.240293</td>
      <td>-2.636852</td>
      <td>0.845372</td>
      <td>-0.252670</td>
      <td>0.935509</td>
      <td>0.931312</td>
      <td>1.807584</td>
      <td>-1.593276</td>
      <td>0.212365</td>
      <td>-0.961358</td>
      <td>0.343816</td>
      <td>0.427158</td>
      <td>-0.726957</td>
      <td>-0.142849</td>
      <td>-1.099036</td>
      <td>-1.171979</td>
      <td>-0.817691</td>
      <td>-0.965264</td>
      <td>-0.402092</td>
      <td>0.163596</td>
      <td>0.390558</td>
      <td>0.213270</td>
      <td>-0.594301</td>
      <td>-0.491219</td>
      <td>0.641704</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 755 columns</p>
</div>




```python
## Implement PCA
pca = PCA()
pca.fit(df_scale)

## Cumulated Variance and extract the number of components which cumulate > 70% of total variance.
cumvar = np.cumsum(pca.explained_variance_ratio_)
cumvar = cumvar[cumvar <= 0.70].tolist()
comp = len(cumvar) + 1 
```


```python
## Explained Variance Ratio
np.cumsum(pca.explained_variance_ratio_)[0:100]
```




    array([0.1080524 , 0.18448808, 0.2208393 , 0.24999421, 0.27572699,
           0.29969785, 0.32238567, 0.34413393, 0.36420684, 0.38171801,
           0.39785179, 0.41220401, 0.42570554, 0.43867622, 0.45123161,
           0.46285066, 0.47437718, 0.48568284, 0.49614924, 0.50629368,
           0.51617265, 0.524697  , 0.53273209, 0.54033421, 0.54755404,
           0.55414613, 0.56066346, 0.56672783, 0.57257453, 0.57812457,
           0.58359364, 0.58882496, 0.59399115, 0.59894491, 0.60372394,
           0.60836552, 0.61293231, 0.61739253, 0.62177469, 0.62596953,
           0.63004576, 0.63402986, 0.6379555 , 0.6418094 , 0.64552984,
           0.64913323, 0.65267055, 0.65606222, 0.65945144, 0.66275489,
           0.66603992, 0.66922337, 0.67234985, 0.67545908, 0.67851901,
           0.68148124, 0.68439171, 0.68725278, 0.69009632, 0.69288388,
           0.69564991, 0.69837672, 0.70108335, 0.7037162 , 0.70633711,
           0.7088601 , 0.71135108, 0.71380461, 0.71622181, 0.71862291,
           0.72097241, 0.72331012, 0.72561747, 0.72789759, 0.73017325,
           0.7324053 , 0.73460497, 0.73678483, 0.73891506, 0.74103372,
           0.7431152 , 0.74518215, 0.74721793, 0.74924752, 0.75125049,
           0.75324029, 0.75521278, 0.75716285, 0.75909903, 0.76098021,
           0.76285866, 0.76471669, 0.7665544 , 0.76837804, 0.77019112,
           0.7719881 , 0.77376491, 0.77552028, 0.77726693, 0.7789893 ])



The first 100 components found are observed. Fixed the minimum threshold of variance that we want to maintain, that is 70%, we see the last component to maintain to preserve this threshold of variability.


```python
## Result
print('70% of variance is caught by', comp, 'components')
```

    70% of variance is caught by 63 components
    


```python
## Plot of Cumulative Variance
figure(figsize = (12, 6), dpi = 80)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axvline(comp, color = 'red');
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.xlim((0,100))
print('As we can see, the 70% of variance is caught by', comp, 'components.')
```

    As we can see, the 70% of variance is caught by 63 components.
    


    
![png](output_89_1.png)
    


### 2.3 - Clustering

### 2.3.1 Implementation of K-Means

#### PseudoCode

**Algorithm**: Lloyd’s approximate K-Means

**input** : X, K, an initial random partition $C_1^{0}, C_2^{0}, ..., C_k^{0}$

Set: $s <- 0$

Set: $itera <- TRUE$

while $(itera == TRUE)$ **do**

  - Step A: Calculates the group centers from the previous partition.
  
  - Step B : creates a new partition by assigning a point i to the group with the nearest center.
  
  - Convergence check
  
  - itera <- FALSE if  $C_k^{(s+1)} = C_k^{s} \forall k = 1, 2, ..., K$

end

**return** final partition $C_1^{(s+1)}, C_2^{(s+1)}, ..., C_k^{(s+1)}.$ 

Extract Score: $Z = X \times ϕ$.


```python
## Extract Score
Z = pca.transform(df_scale)
Z = pd.DataFrame(Z)
## We extract the projection of X in the principal components.

## Extract first 64 cocmponents.
Z = Z.iloc[0:, :comp]
```


```python
## Check
Z.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.064849</td>
      <td>-2.831589</td>
      <td>-5.729978</td>
      <td>4.316744</td>
      <td>3.129221</td>
      <td>2.052358</td>
      <td>6.770295</td>
      <td>-0.866999</td>
      <td>-2.207433</td>
      <td>4.398128</td>
      <td>-4.815142</td>
      <td>1.518511</td>
      <td>-4.292408</td>
      <td>5.288776</td>
      <td>-0.313838</td>
      <td>2.191539</td>
      <td>-0.964954</td>
      <td>-0.942773</td>
      <td>1.600828</td>
      <td>0.739939</td>
      <td>-2.261945</td>
      <td>-0.835331</td>
      <td>-1.989693</td>
      <td>1.663556</td>
      <td>0.245751</td>
      <td>-3.182379</td>
      <td>0.320655</td>
      <td>0.957173</td>
      <td>3.210551</td>
      <td>-0.600194</td>
      <td>1.330839</td>
      <td>-1.316388</td>
      <td>-0.002667</td>
      <td>-0.698462</td>
      <td>-1.729454</td>
      <td>1.856724</td>
      <td>-1.506390</td>
      <td>2.377169</td>
      <td>-0.660971</td>
      <td>-0.340542</td>
      <td>-3.263969</td>
      <td>-0.013245</td>
      <td>-1.443718</td>
      <td>-0.717360</td>
      <td>1.682789</td>
      <td>0.489983</td>
      <td>2.042114</td>
      <td>1.332147</td>
      <td>-0.296035</td>
      <td>0.473354</td>
      <td>0.611951</td>
      <td>0.937544</td>
      <td>-0.104555</td>
      <td>1.616836</td>
      <td>-1.153070</td>
      <td>-1.585754</td>
      <td>-0.220216</td>
      <td>-0.528155</td>
      <td>-1.146545</td>
      <td>-2.422353</td>
      <td>1.069767</td>
      <td>0.277392</td>
      <td>-0.295876</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.775029</td>
      <td>-5.416029</td>
      <td>-5.295198</td>
      <td>-1.580781</td>
      <td>-3.646130</td>
      <td>2.632201</td>
      <td>-1.540514</td>
      <td>5.934789</td>
      <td>-0.102422</td>
      <td>4.215868</td>
      <td>-2.155764</td>
      <td>2.505115</td>
      <td>-0.911807</td>
      <td>3.826839</td>
      <td>1.978222</td>
      <td>1.414908</td>
      <td>-1.844117</td>
      <td>-3.983110</td>
      <td>-0.612405</td>
      <td>1.483711</td>
      <td>0.388609</td>
      <td>-1.442001</td>
      <td>-0.015714</td>
      <td>3.585421</td>
      <td>0.389612</td>
      <td>-1.065766</td>
      <td>1.388909</td>
      <td>-0.447937</td>
      <td>1.236075</td>
      <td>0.068605</td>
      <td>2.067720</td>
      <td>-0.540475</td>
      <td>-0.592127</td>
      <td>0.728173</td>
      <td>-3.090387</td>
      <td>-0.439931</td>
      <td>-0.979652</td>
      <td>1.361186</td>
      <td>2.549790</td>
      <td>2.357500</td>
      <td>-0.316553</td>
      <td>-0.861418</td>
      <td>-0.851712</td>
      <td>-1.466467</td>
      <td>3.362865</td>
      <td>0.812371</td>
      <td>-0.262717</td>
      <td>0.972362</td>
      <td>0.592669</td>
      <td>-1.066389</td>
      <td>-1.160548</td>
      <td>0.454529</td>
      <td>-1.079982</td>
      <td>-0.083025</td>
      <td>-2.599944</td>
      <td>-1.170376</td>
      <td>-0.195097</td>
      <td>-1.238873</td>
      <td>-0.149774</td>
      <td>-0.507287</td>
      <td>-0.650328</td>
      <td>0.893198</td>
      <td>0.411697</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.648080</td>
      <td>-4.175493</td>
      <td>-4.569957</td>
      <td>-0.041332</td>
      <td>-2.860837</td>
      <td>2.094845</td>
      <td>-1.275281</td>
      <td>-0.315749</td>
      <td>-0.100084</td>
      <td>1.836875</td>
      <td>-2.051694</td>
      <td>3.126571</td>
      <td>-3.259031</td>
      <td>2.596994</td>
      <td>0.170699</td>
      <td>2.231499</td>
      <td>-0.516802</td>
      <td>-3.101971</td>
      <td>2.411321</td>
      <td>-1.503274</td>
      <td>-2.913391</td>
      <td>-1.394451</td>
      <td>2.218486</td>
      <td>2.838376</td>
      <td>-0.615166</td>
      <td>-1.928976</td>
      <td>1.820476</td>
      <td>-0.247708</td>
      <td>0.094223</td>
      <td>-0.150519</td>
      <td>1.272948</td>
      <td>2.772211</td>
      <td>1.085103</td>
      <td>-2.039939</td>
      <td>-0.312135</td>
      <td>0.725741</td>
      <td>-1.641434</td>
      <td>0.397542</td>
      <td>2.004948</td>
      <td>1.564528</td>
      <td>-1.373810</td>
      <td>-0.345909</td>
      <td>-1.141654</td>
      <td>-1.377168</td>
      <td>-0.650188</td>
      <td>1.458724</td>
      <td>0.781534</td>
      <td>-0.027286</td>
      <td>-2.021507</td>
      <td>0.442614</td>
      <td>-0.210446</td>
      <td>2.216602</td>
      <td>-1.473105</td>
      <td>2.036536</td>
      <td>-3.116759</td>
      <td>-0.907232</td>
      <td>-2.361728</td>
      <td>0.615146</td>
      <td>-0.838890</td>
      <td>-1.202350</td>
      <td>-1.199162</td>
      <td>1.506561</td>
      <td>1.375364</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.234913</td>
      <td>3.775000</td>
      <td>-5.545841</td>
      <td>0.699968</td>
      <td>-1.413928</td>
      <td>0.426163</td>
      <td>-1.814266</td>
      <td>4.046189</td>
      <td>-0.539085</td>
      <td>4.977383</td>
      <td>-1.920963</td>
      <td>6.719147</td>
      <td>2.054402</td>
      <td>0.310927</td>
      <td>-1.547769</td>
      <td>-4.132607</td>
      <td>-0.089768</td>
      <td>-2.731083</td>
      <td>3.527946</td>
      <td>-1.013561</td>
      <td>-2.586001</td>
      <td>3.323053</td>
      <td>-2.332978</td>
      <td>0.005254</td>
      <td>2.041952</td>
      <td>-0.829880</td>
      <td>-1.756317</td>
      <td>0.793297</td>
      <td>0.194241</td>
      <td>2.571039</td>
      <td>-1.735934</td>
      <td>-2.696700</td>
      <td>2.455623</td>
      <td>-0.591208</td>
      <td>0.274002</td>
      <td>-5.769041</td>
      <td>-1.246138</td>
      <td>-0.044766</td>
      <td>-0.178684</td>
      <td>1.059918</td>
      <td>-1.444050</td>
      <td>-0.220837</td>
      <td>-0.434316</td>
      <td>-2.658855</td>
      <td>2.505102</td>
      <td>1.622691</td>
      <td>0.560247</td>
      <td>-0.232789</td>
      <td>-4.291800</td>
      <td>-0.568013</td>
      <td>2.583597</td>
      <td>0.486105</td>
      <td>1.730453</td>
      <td>-0.677343</td>
      <td>-0.684793</td>
      <td>-0.127586</td>
      <td>1.130949</td>
      <td>-0.471296</td>
      <td>0.406995</td>
      <td>-0.636077</td>
      <td>0.709149</td>
      <td>0.194866</td>
      <td>1.737456</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.306618</td>
      <td>-9.465927</td>
      <td>-4.280237</td>
      <td>0.747145</td>
      <td>-3.842896</td>
      <td>-1.208390</td>
      <td>-2.934347</td>
      <td>-2.197162</td>
      <td>-0.511378</td>
      <td>2.553976</td>
      <td>-0.510711</td>
      <td>1.405750</td>
      <td>-1.352031</td>
      <td>1.074685</td>
      <td>-0.928036</td>
      <td>1.833188</td>
      <td>2.295927</td>
      <td>-1.669374</td>
      <td>0.631459</td>
      <td>2.418053</td>
      <td>-0.078134</td>
      <td>-2.432385</td>
      <td>0.269018</td>
      <td>0.174332</td>
      <td>-1.215254</td>
      <td>-2.270264</td>
      <td>2.394836</td>
      <td>0.326568</td>
      <td>1.607412</td>
      <td>1.108830</td>
      <td>-1.255134</td>
      <td>-0.327092</td>
      <td>0.876635</td>
      <td>-2.208845</td>
      <td>-0.654325</td>
      <td>0.566157</td>
      <td>-1.348874</td>
      <td>-1.198655</td>
      <td>2.135779</td>
      <td>-0.575339</td>
      <td>-1.347962</td>
      <td>-0.427463</td>
      <td>-0.607911</td>
      <td>0.337198</td>
      <td>1.477528</td>
      <td>0.253536</td>
      <td>-0.247333</td>
      <td>-0.372204</td>
      <td>-0.907106</td>
      <td>-0.152948</td>
      <td>0.235667</td>
      <td>-0.025401</td>
      <td>0.749870</td>
      <td>0.541306</td>
      <td>-0.074447</td>
      <td>-2.240163</td>
      <td>-0.742984</td>
      <td>-2.546209</td>
      <td>0.703664</td>
      <td>-2.418568</td>
      <td>-2.549477</td>
      <td>2.344405</td>
      <td>0.475233</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Implementation of K-Means
def kmeans(X, k, n_iter = 100):
  if isinstance(X, pd.DataFrame):X = X.values
  first = first_cluster(X, k, n_iter)
  clusters, centroids = make_clusters(X, k, first, n_iter)
  return clusters, centroids

def first_cluster(X, k, n_iter):
  idx = np.random.choice(len(X), k, replace=False)
  centers = X[idx, :]
  cluster = np.argmin(distance.cdist(X, centers, 'euclidean'), axis = 1)
  return cluster

def make_clusters(X, k, first_cluster, n_iter):
  cluster = first_cluster
  for _ in range(n_iter):
    centers = []
    for i in range(k):
      centers.append(X[cluster == i,:].mean(axis = 0))
    centers = np.vstack(centers)
    tmp = np.argmin(distance.cdist(X, centers, 'euclidean'), axis = 1)
    if np.array_equal(cluster,tmp):
      break
    cluster = tmp
    return cluster, centers 
```


```python
## Check for 3 cluster
cluster, centroids = kmeans(Z, 3, 100)
```


```python
## Plot Clustering
Z = Z.to_numpy()
figure(figsize = (12, 6), dpi = 80)
u_labels = np.unique(cluster)
for i in u_labels:
    plt.scatter(Z[cluster == i , 0] , Z[cluster == i , 1] , label = i)
plt.legend()
plt.show()
```


    
![png](output_98_0.png)
    


### 2.3.2 Find an optimal number of clusters
#### **Validation** 
Clustering is an unsupervised task. Ex-post we cannot compare results with **ground-truth*. However, we need tools that somehow give us a measure of the quality of clustering.

#### **Internal Validation**
It refers to the fact that the validation of a partition is done using only the information contained in the input data set.

#### **Choice of number of groups K**
- In some methods it is seen as an estimation problem, this presupposes the existence of a model that defines a *true* $K$.
- In many applications different K values could give a *good* representation of the population.
- the choice of K is a problem of validation rather than estimation.

> Elbow Method



```python
## Cluster for different k
k = 30

clusters = []
centroids = []

for k in range(2, k+1):
  cluster, centroid = kmeans(Z, k)
  clusters.append(cluster)
  centroids.append(centroid)
```


```python
## WSS
WSS = []
for i in centroids:
  within = np.min(distance.cdist(Z, i, 'euclidean'), axis = 1)
  within = within ** 2
  WSS.append(sum(within))
```


```python
## Scree Plot
x = []
for i in range(2, k+1):
  x.append(i)
plt.figure(figsize = (12, 6), dpi = 80)
plt.plot(x, WSS, 'o-', linewidth = 1.5, color = 'red')
plt.title('Scree Plot')
plt.xlim(1, 31)
plt.show()
```


    
![png](output_103_0.png)
    


By plotting the **WSS** for each K value and using the **Elbow method**, you notice that there is not a single optimal value of k. Surely an important structure is suggested for a value of *k* equal to *8* and for a value of *k* equal to *12.*

> Silhouette Width
##### **Inputs**:
- Vector with cluster labels, dissimilarity matrix.
##### **Philosophy**:
- In a good clustering each point is well connected to its cluster, while it is poorly connected to other clusters.
##### **Silhouette Width di i**:
- $s(i) = \frac{b(i) - a(i)}{\max({a(i),b(i)})}$, with $s(i) \in [-1, 1]$
- $s(i) \approx 1: i$ is well accommodated in its cluster

- $s(i) \approx -1: i$ would be better in another cluster

- $s(i) \approx 0: i$ is a transition region between two clusters
##### **Average Silhouette Width**:
- $ASW = \frac{1}{n} \sum_{i=1}^n s(i)$

**Note**: we want clustering with a high ASW.


```python
## Implementation
def calculate_silhouette(D_copy, labels, centers, k):
    score =[]
    for c in range(k):
        inter_c = D_copy[D_copy.labels == c]
        for i in range(len(inter_c)):
            a = distance.cdist(inter_c.iloc[i:i+1, :-1], D_copy[D_copy.labels == c].iloc[:,:-1], 'euclidean')
            a_i = np.sum(a) / len(inter_c)

            dist = distance.cdist(inter_c.iloc[i:i+1,:-1], centers, 'euclidean')

            next = np.argsort(dist)[0][1]

            b = distance.cdist(inter_c.iloc[i:i+1,:-1], D_copy[D_copy.labels == next].iloc[:,:-1], 'euclidean')
            b_i = np.mean(b)

            s = (b_i - a_i) / max(a_i, b_i)
            score.append(s)

    return np.mean(score)

def silhouette(D, labels, centers):
  D_copy = D.copy()
  res = []
  for k in range(len(labels)):
    D_copy['labels'] = labels[k]
    res.append(calculate_silhouette(D_copy, labels[k], centers[k], k+2))
  return res
```


```python
## Score
score = silhouette(Z, clusters, centroids)
```


```python
## Scree Plot
x = []
for i in range(2, k+1):
  x.append(i)
plt.figure(figsize = (12, 6), dpi = 80)
plt.plot(x, score, 'o-', linewidth = 1.5, color = 'red')
plt.title('Scree Plot')
plt.xlim(1, 31)
plt.show()
```


    
![png](output_108_0.png)
    


### 2.3.3 - Run the algorithm on the data that you got from the dimensionality reduction


```python
## Run the algorithm on the data that you got from the dimensionality reduction.
## K = 8
k_8 = 8
cluster_8, centroids_8 = kmeans(Z, k_8, 100)
```


```python
Z = pd.DataFrame(Z)
```


```python
## Plot Clustering
Z = Z.to_numpy()
figure(figsize = (12, 6), dpi = 80)
u_labels = np.unique(cluster_8)
for i in u_labels:
    plt.scatter(Z[cluster_8 == i , 0] , Z[cluster_8 == i , 1] , label = i)
plt.legend()
plt.show()
```


    
![png](output_112_0.png)
    



```python
## Run the algorithm on the data that you got from the dimensionality reduction.
## K = 12
k_optimal = 12
cluster_12, centroids_12 = kmeans(Z, k_optimal, 100)
```


```python
Z = pd.DataFrame(Z)
```


```python
## Plot Clustering
Z = Z.to_numpy()
figure(figsize = (12, 6), dpi = 80)
u_labels = np.unique(cluster_12)
for i in u_labels:
    plt.scatter(Z[cluster_12 == i , 0] , Z[cluster_12 == i , 1] , label = i)
plt.legend()
plt.show()
```


    
![png](output_115_0.png)
    


### 2.3.4 - K-Means++

Both **K-means** and **K-means++** are clustering methods which comes under unsupervised learning. The main difference between the two algorithms lies in:

- the selection of the centroids around which the clustering takes place.
- k means++ removes the drawback of K means which is it is dependent on initialization of centroid.


```python
## Metric = Distortion, which computes the sum of squared distances from each point to its assigned center. 
figure(figsize = (12, 6), dpi = 80)
model = KMeans(init = 'k-means++')
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'distortion', timings = False)
visualizer.fit(Z)
visualizer.show()
```


    
![png](output_118_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94d494d690>



By using **k-means++** and following the **Elbow** method, the optimal **value** is *k = 11* with a distortion score of *4994131.860*. In this sense, the choice of the optimal k does not differ much from that obtained with the kmeans from scratch.


```python
## Metric = Silhouette, which score calculates the mean Silhouette Coefficient of all samples.
figure(figsize = (12, 6), dpi = 80)
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'silhouette', timings = False)
visualizer.fit(Z)
visualizer.show()
```


    
![png](output_120_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94d3f22290>




```python
## Metric = Calinski-Harabaszion, which score computes the ratio of dispersion between and within clusters.
figure(figsize = (12, 6), dpi = 80)
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'calinski_harabasz', timings = False)
visualizer.fit(Z)
visualizer.show()
```


    
![png](output_121_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94d4845cd0>



Using the *calinski_harabasz* and *silhouette* metrics, the optimal number of *k* is equal to *2*. A different result that may depend on the different way these scores are calculated. Nevertheless, the value of the *silhouette* is very low and therefore does not lead us to choose *k = 2* as the optimal value.

Even if the choice of the number of components is increased by **80%** of the variance, the choice of the number of k looking at the *calinski_harabasz* and *silhouette* metrics, does not change, while looking at the *distortion score*, the optimal *k* value is *11*.


```python
## Implement PCA (80% of variance)
pca2 = PCA()
pca2.fit(df_scale)

## Cumulated Variance and extract the number of components which cumulate > 70% of total variance.
cumvar2 = np.cumsum(pca2.explained_variance_ratio_)
cumvar2 = cumvar2[cumvar2 <= 0.80].tolist()
comp2 = len(cumvar2) + 1 
```


```python
## Result
print('80% of variance is caught by', comp2, 'components')
```

    80% of variance is caught by 113 components
    


```python
## Extract Score
Z2 = pca.transform(df_scale)
Z2 = pd.DataFrame(Z2)
## We extract the projection of X in the principal components.

## Extract first 113 cocmponents.
Z2 = Z2.iloc[0:, :comp2]
```


```python
## Check
Z2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>...</th>
      <th>73</th>
      <th>74</th>
      <th>75</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>85</th>
      <th>86</th>
      <th>87</th>
      <th>88</th>
      <th>89</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
      <th>100</th>
      <th>101</th>
      <th>102</th>
      <th>103</th>
      <th>104</th>
      <th>105</th>
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.064849</td>
      <td>-2.831589</td>
      <td>-5.729978</td>
      <td>4.316744</td>
      <td>3.129221</td>
      <td>2.052358</td>
      <td>6.770295</td>
      <td>-0.866999</td>
      <td>-2.207433</td>
      <td>4.398128</td>
      <td>-4.815142</td>
      <td>1.518511</td>
      <td>-4.292408</td>
      <td>5.288776</td>
      <td>-0.313838</td>
      <td>2.191539</td>
      <td>-0.964954</td>
      <td>-0.942773</td>
      <td>1.600828</td>
      <td>0.739939</td>
      <td>-2.261945</td>
      <td>-0.835331</td>
      <td>-1.989693</td>
      <td>1.663556</td>
      <td>0.245751</td>
      <td>-3.182379</td>
      <td>0.320655</td>
      <td>0.957173</td>
      <td>3.210551</td>
      <td>-0.600194</td>
      <td>1.330839</td>
      <td>-1.316388</td>
      <td>-0.002667</td>
      <td>-0.698462</td>
      <td>-1.729454</td>
      <td>1.856724</td>
      <td>-1.506390</td>
      <td>2.377169</td>
      <td>-0.660971</td>
      <td>-0.340542</td>
      <td>...</td>
      <td>0.554949</td>
      <td>1.094093</td>
      <td>-1.413513</td>
      <td>-2.319282</td>
      <td>1.031345</td>
      <td>-1.220156</td>
      <td>0.915555</td>
      <td>-0.901345</td>
      <td>-1.106523</td>
      <td>-1.304501</td>
      <td>0.943831</td>
      <td>0.766558</td>
      <td>-0.952965</td>
      <td>-1.621375</td>
      <td>-0.548516</td>
      <td>-1.646250</td>
      <td>2.186895</td>
      <td>1.180284</td>
      <td>-0.085037</td>
      <td>-0.562896</td>
      <td>-0.379446</td>
      <td>2.065050</td>
      <td>-1.221707</td>
      <td>0.207919</td>
      <td>-0.517246</td>
      <td>-0.145105</td>
      <td>-0.134096</td>
      <td>-0.043471</td>
      <td>0.715065</td>
      <td>0.208679</td>
      <td>0.555634</td>
      <td>-0.367842</td>
      <td>-0.298764</td>
      <td>-1.643241</td>
      <td>-0.158727</td>
      <td>0.342162</td>
      <td>-0.065645</td>
      <td>1.406162</td>
      <td>0.445772</td>
      <td>1.820688</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.775029</td>
      <td>-5.416029</td>
      <td>-5.295198</td>
      <td>-1.580781</td>
      <td>-3.646130</td>
      <td>2.632201</td>
      <td>-1.540514</td>
      <td>5.934789</td>
      <td>-0.102422</td>
      <td>4.215868</td>
      <td>-2.155764</td>
      <td>2.505115</td>
      <td>-0.911807</td>
      <td>3.826839</td>
      <td>1.978222</td>
      <td>1.414908</td>
      <td>-1.844117</td>
      <td>-3.983110</td>
      <td>-0.612405</td>
      <td>1.483711</td>
      <td>0.388609</td>
      <td>-1.442001</td>
      <td>-0.015714</td>
      <td>3.585421</td>
      <td>0.389612</td>
      <td>-1.065766</td>
      <td>1.388909</td>
      <td>-0.447937</td>
      <td>1.236075</td>
      <td>0.068605</td>
      <td>2.067720</td>
      <td>-0.540475</td>
      <td>-0.592127</td>
      <td>0.728173</td>
      <td>-3.090387</td>
      <td>-0.439931</td>
      <td>-0.979652</td>
      <td>1.361186</td>
      <td>2.549790</td>
      <td>2.357500</td>
      <td>...</td>
      <td>0.365660</td>
      <td>-0.739060</td>
      <td>1.169117</td>
      <td>-0.124618</td>
      <td>-0.471960</td>
      <td>0.415834</td>
      <td>-0.652740</td>
      <td>-1.644239</td>
      <td>-1.507742</td>
      <td>0.564391</td>
      <td>0.127124</td>
      <td>0.697295</td>
      <td>-0.312142</td>
      <td>0.147377</td>
      <td>-0.344970</td>
      <td>2.526019</td>
      <td>-1.084969</td>
      <td>-0.486964</td>
      <td>-1.093041</td>
      <td>-0.265865</td>
      <td>0.829146</td>
      <td>-0.546494</td>
      <td>-1.207006</td>
      <td>0.674351</td>
      <td>-1.070685</td>
      <td>-1.082955</td>
      <td>-0.834988</td>
      <td>-0.139010</td>
      <td>0.373146</td>
      <td>-0.421449</td>
      <td>0.437901</td>
      <td>-0.883664</td>
      <td>-0.383767</td>
      <td>0.399235</td>
      <td>-0.101801</td>
      <td>1.125598</td>
      <td>-0.276657</td>
      <td>-0.437484</td>
      <td>1.395319</td>
      <td>0.072270</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.648080</td>
      <td>-4.175493</td>
      <td>-4.569957</td>
      <td>-0.041332</td>
      <td>-2.860837</td>
      <td>2.094845</td>
      <td>-1.275281</td>
      <td>-0.315749</td>
      <td>-0.100084</td>
      <td>1.836875</td>
      <td>-2.051694</td>
      <td>3.126571</td>
      <td>-3.259031</td>
      <td>2.596994</td>
      <td>0.170699</td>
      <td>2.231499</td>
      <td>-0.516802</td>
      <td>-3.101971</td>
      <td>2.411321</td>
      <td>-1.503274</td>
      <td>-2.913391</td>
      <td>-1.394451</td>
      <td>2.218486</td>
      <td>2.838376</td>
      <td>-0.615166</td>
      <td>-1.928976</td>
      <td>1.820476</td>
      <td>-0.247708</td>
      <td>0.094223</td>
      <td>-0.150519</td>
      <td>1.272948</td>
      <td>2.772211</td>
      <td>1.085103</td>
      <td>-2.039939</td>
      <td>-0.312135</td>
      <td>0.725741</td>
      <td>-1.641434</td>
      <td>0.397542</td>
      <td>2.004948</td>
      <td>1.564528</td>
      <td>...</td>
      <td>0.450113</td>
      <td>1.579316</td>
      <td>0.476297</td>
      <td>-0.899068</td>
      <td>1.209743</td>
      <td>0.771750</td>
      <td>-0.373718</td>
      <td>-0.004293</td>
      <td>0.813592</td>
      <td>-1.222102</td>
      <td>-0.390276</td>
      <td>-0.400611</td>
      <td>-0.475126</td>
      <td>-1.522752</td>
      <td>-1.783367</td>
      <td>-1.893461</td>
      <td>0.917226</td>
      <td>1.263655</td>
      <td>-0.356338</td>
      <td>-1.243684</td>
      <td>0.179432</td>
      <td>-0.270468</td>
      <td>-0.123128</td>
      <td>-0.403112</td>
      <td>0.503911</td>
      <td>-0.151257</td>
      <td>-0.486149</td>
      <td>1.129529</td>
      <td>1.119802</td>
      <td>0.364968</td>
      <td>-0.478926</td>
      <td>-0.034527</td>
      <td>0.156130</td>
      <td>-1.581817</td>
      <td>0.348820</td>
      <td>-0.549895</td>
      <td>-0.021196</td>
      <td>0.014888</td>
      <td>1.893457</td>
      <td>-0.381760</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.234913</td>
      <td>3.775000</td>
      <td>-5.545841</td>
      <td>0.699968</td>
      <td>-1.413928</td>
      <td>0.426163</td>
      <td>-1.814266</td>
      <td>4.046189</td>
      <td>-0.539085</td>
      <td>4.977383</td>
      <td>-1.920963</td>
      <td>6.719147</td>
      <td>2.054402</td>
      <td>0.310927</td>
      <td>-1.547769</td>
      <td>-4.132607</td>
      <td>-0.089768</td>
      <td>-2.731083</td>
      <td>3.527946</td>
      <td>-1.013561</td>
      <td>-2.586001</td>
      <td>3.323053</td>
      <td>-2.332978</td>
      <td>0.005254</td>
      <td>2.041952</td>
      <td>-0.829880</td>
      <td>-1.756317</td>
      <td>0.793297</td>
      <td>0.194241</td>
      <td>2.571039</td>
      <td>-1.735934</td>
      <td>-2.696700</td>
      <td>2.455623</td>
      <td>-0.591208</td>
      <td>0.274002</td>
      <td>-5.769041</td>
      <td>-1.246138</td>
      <td>-0.044766</td>
      <td>-0.178684</td>
      <td>1.059918</td>
      <td>...</td>
      <td>0.866091</td>
      <td>-0.219413</td>
      <td>1.123884</td>
      <td>-1.900957</td>
      <td>-2.514749</td>
      <td>0.065324</td>
      <td>0.681720</td>
      <td>0.676693</td>
      <td>0.002423</td>
      <td>2.063892</td>
      <td>0.479649</td>
      <td>-1.620369</td>
      <td>-1.398610</td>
      <td>1.224292</td>
      <td>-0.530423</td>
      <td>0.131420</td>
      <td>-2.593367</td>
      <td>-2.026331</td>
      <td>1.277367</td>
      <td>-1.532705</td>
      <td>1.771878</td>
      <td>-0.566439</td>
      <td>0.030420</td>
      <td>-0.549311</td>
      <td>-0.342397</td>
      <td>1.360026</td>
      <td>1.257243</td>
      <td>-0.573967</td>
      <td>0.751964</td>
      <td>1.156901</td>
      <td>-0.905197</td>
      <td>3.374343</td>
      <td>-0.854752</td>
      <td>3.132372</td>
      <td>0.189920</td>
      <td>-0.620770</td>
      <td>0.238119</td>
      <td>0.512319</td>
      <td>2.008394</td>
      <td>-0.099731</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.306618</td>
      <td>-9.465927</td>
      <td>-4.280237</td>
      <td>0.747145</td>
      <td>-3.842896</td>
      <td>-1.208390</td>
      <td>-2.934347</td>
      <td>-2.197162</td>
      <td>-0.511378</td>
      <td>2.553976</td>
      <td>-0.510711</td>
      <td>1.405750</td>
      <td>-1.352031</td>
      <td>1.074685</td>
      <td>-0.928036</td>
      <td>1.833188</td>
      <td>2.295927</td>
      <td>-1.669374</td>
      <td>0.631459</td>
      <td>2.418053</td>
      <td>-0.078134</td>
      <td>-2.432385</td>
      <td>0.269018</td>
      <td>0.174332</td>
      <td>-1.215254</td>
      <td>-2.270264</td>
      <td>2.394836</td>
      <td>0.326568</td>
      <td>1.607412</td>
      <td>1.108830</td>
      <td>-1.255134</td>
      <td>-0.327092</td>
      <td>0.876635</td>
      <td>-2.208845</td>
      <td>-0.654325</td>
      <td>0.566157</td>
      <td>-1.348874</td>
      <td>-1.198655</td>
      <td>2.135779</td>
      <td>-0.575339</td>
      <td>...</td>
      <td>-1.183345</td>
      <td>0.063648</td>
      <td>0.086958</td>
      <td>0.140494</td>
      <td>-0.004492</td>
      <td>1.767436</td>
      <td>-0.938485</td>
      <td>-1.505320</td>
      <td>-0.119704</td>
      <td>0.427739</td>
      <td>-0.545360</td>
      <td>-0.456394</td>
      <td>1.209458</td>
      <td>-1.289385</td>
      <td>0.472174</td>
      <td>0.868890</td>
      <td>-0.921493</td>
      <td>-0.023370</td>
      <td>-0.701765</td>
      <td>-0.127518</td>
      <td>1.633166</td>
      <td>0.504640</td>
      <td>-0.465858</td>
      <td>1.281056</td>
      <td>0.530979</td>
      <td>-0.625525</td>
      <td>0.119486</td>
      <td>-0.152666</td>
      <td>0.315414</td>
      <td>-0.794958</td>
      <td>0.396001</td>
      <td>-1.176692</td>
      <td>0.419268</td>
      <td>-0.721829</td>
      <td>-0.055010</td>
      <td>1.020955</td>
      <td>0.979382</td>
      <td>0.055693</td>
      <td>0.164161</td>
      <td>-0.501732</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 113 columns</p>
</div>




```python
## Metric = Distortion, which computes the sum of squared distances from each point to its assigned center. 
figure(figsize = (12, 6), dpi = 80)
model = KMeans(init = 'k-means++')
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'distortion', timings = False)
visualizer.fit(Z2)
visualizer.show()
```


    
![png](output_128_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94ff499c50>



Even using the first *j* main components that cumulate *80% of variance*, the choice of the optimal *k* does not differ much from the choice defined previously, both with k*kmeans++* and with *kmeans from scratch*.


```python
## Metric = Silhouette, which score calculates the mean Silhouette Coefficient of all samples.
figure(figsize = (12, 6), dpi = 80)
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'silhouette', timings = False)
visualizer.fit(Z2)
visualizer.show()
```


    
![png](output_130_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94d3ec3390>




```python
## Metric = Calinski-Harabaszion, which score computes the ratio of dispersion between and within clusters.
figure(figsize = (12, 6), dpi = 80)
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'calinski_harabasz', timings = False)
visualizer.fit(Z2)
visualizer.show()
```


    
![png](output_131_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94d3dde110>



### 2.4 - Analysing your results


```python
## Add the list of cluster to dataset 
Z = pd.DataFrame(Z)
Z['cluster_12'] = cluster_12 
```


```python
Z.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>cluster_12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.064849</td>
      <td>-2.831589</td>
      <td>-5.729978</td>
      <td>4.316744</td>
      <td>3.129221</td>
      <td>2.052358</td>
      <td>6.770295</td>
      <td>-0.866999</td>
      <td>-2.207433</td>
      <td>4.398128</td>
      <td>-4.815142</td>
      <td>1.518511</td>
      <td>-4.292408</td>
      <td>5.288776</td>
      <td>-0.313838</td>
      <td>2.191539</td>
      <td>-0.964954</td>
      <td>-0.942773</td>
      <td>1.600828</td>
      <td>0.739939</td>
      <td>-2.261945</td>
      <td>-0.835331</td>
      <td>-1.989693</td>
      <td>1.663556</td>
      <td>0.245751</td>
      <td>-3.182379</td>
      <td>0.320655</td>
      <td>0.957173</td>
      <td>3.210551</td>
      <td>-0.600194</td>
      <td>1.330839</td>
      <td>-1.316388</td>
      <td>-0.002667</td>
      <td>-0.698462</td>
      <td>-1.729454</td>
      <td>1.856724</td>
      <td>-1.506390</td>
      <td>2.377169</td>
      <td>-0.660971</td>
      <td>-0.340542</td>
      <td>-3.263969</td>
      <td>-0.013245</td>
      <td>-1.443718</td>
      <td>-0.717360</td>
      <td>1.682789</td>
      <td>0.489983</td>
      <td>2.042114</td>
      <td>1.332147</td>
      <td>-0.296035</td>
      <td>0.473354</td>
      <td>0.611951</td>
      <td>0.937544</td>
      <td>-0.104555</td>
      <td>1.616836</td>
      <td>-1.153070</td>
      <td>-1.585754</td>
      <td>-0.220216</td>
      <td>-0.528155</td>
      <td>-1.146545</td>
      <td>-2.422353</td>
      <td>1.069767</td>
      <td>0.277392</td>
      <td>-0.295876</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.775029</td>
      <td>-5.416029</td>
      <td>-5.295198</td>
      <td>-1.580781</td>
      <td>-3.646130</td>
      <td>2.632201</td>
      <td>-1.540514</td>
      <td>5.934789</td>
      <td>-0.102422</td>
      <td>4.215868</td>
      <td>-2.155764</td>
      <td>2.505115</td>
      <td>-0.911807</td>
      <td>3.826839</td>
      <td>1.978222</td>
      <td>1.414908</td>
      <td>-1.844117</td>
      <td>-3.983110</td>
      <td>-0.612405</td>
      <td>1.483711</td>
      <td>0.388609</td>
      <td>-1.442001</td>
      <td>-0.015714</td>
      <td>3.585421</td>
      <td>0.389612</td>
      <td>-1.065766</td>
      <td>1.388909</td>
      <td>-0.447937</td>
      <td>1.236075</td>
      <td>0.068605</td>
      <td>2.067720</td>
      <td>-0.540475</td>
      <td>-0.592127</td>
      <td>0.728173</td>
      <td>-3.090387</td>
      <td>-0.439931</td>
      <td>-0.979652</td>
      <td>1.361186</td>
      <td>2.549790</td>
      <td>2.357500</td>
      <td>-0.316553</td>
      <td>-0.861418</td>
      <td>-0.851712</td>
      <td>-1.466467</td>
      <td>3.362865</td>
      <td>0.812371</td>
      <td>-0.262717</td>
      <td>0.972362</td>
      <td>0.592669</td>
      <td>-1.066389</td>
      <td>-1.160548</td>
      <td>0.454529</td>
      <td>-1.079982</td>
      <td>-0.083025</td>
      <td>-2.599944</td>
      <td>-1.170376</td>
      <td>-0.195097</td>
      <td>-1.238873</td>
      <td>-0.149774</td>
      <td>-0.507287</td>
      <td>-0.650328</td>
      <td>0.893198</td>
      <td>0.411697</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.648080</td>
      <td>-4.175493</td>
      <td>-4.569957</td>
      <td>-0.041332</td>
      <td>-2.860837</td>
      <td>2.094845</td>
      <td>-1.275281</td>
      <td>-0.315749</td>
      <td>-0.100084</td>
      <td>1.836875</td>
      <td>-2.051694</td>
      <td>3.126571</td>
      <td>-3.259031</td>
      <td>2.596994</td>
      <td>0.170699</td>
      <td>2.231499</td>
      <td>-0.516802</td>
      <td>-3.101971</td>
      <td>2.411321</td>
      <td>-1.503274</td>
      <td>-2.913391</td>
      <td>-1.394451</td>
      <td>2.218486</td>
      <td>2.838376</td>
      <td>-0.615166</td>
      <td>-1.928976</td>
      <td>1.820476</td>
      <td>-0.247708</td>
      <td>0.094223</td>
      <td>-0.150519</td>
      <td>1.272948</td>
      <td>2.772211</td>
      <td>1.085103</td>
      <td>-2.039939</td>
      <td>-0.312135</td>
      <td>0.725741</td>
      <td>-1.641434</td>
      <td>0.397542</td>
      <td>2.004948</td>
      <td>1.564528</td>
      <td>-1.373810</td>
      <td>-0.345909</td>
      <td>-1.141654</td>
      <td>-1.377168</td>
      <td>-0.650188</td>
      <td>1.458724</td>
      <td>0.781534</td>
      <td>-0.027286</td>
      <td>-2.021507</td>
      <td>0.442614</td>
      <td>-0.210446</td>
      <td>2.216602</td>
      <td>-1.473105</td>
      <td>2.036536</td>
      <td>-3.116759</td>
      <td>-0.907232</td>
      <td>-2.361728</td>
      <td>0.615146</td>
      <td>-0.838890</td>
      <td>-1.202350</td>
      <td>-1.199162</td>
      <td>1.506561</td>
      <td>1.375364</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.234913</td>
      <td>3.775000</td>
      <td>-5.545841</td>
      <td>0.699968</td>
      <td>-1.413928</td>
      <td>0.426163</td>
      <td>-1.814266</td>
      <td>4.046189</td>
      <td>-0.539085</td>
      <td>4.977383</td>
      <td>-1.920963</td>
      <td>6.719147</td>
      <td>2.054402</td>
      <td>0.310927</td>
      <td>-1.547769</td>
      <td>-4.132607</td>
      <td>-0.089768</td>
      <td>-2.731083</td>
      <td>3.527946</td>
      <td>-1.013561</td>
      <td>-2.586001</td>
      <td>3.323053</td>
      <td>-2.332978</td>
      <td>0.005254</td>
      <td>2.041952</td>
      <td>-0.829880</td>
      <td>-1.756317</td>
      <td>0.793297</td>
      <td>0.194241</td>
      <td>2.571039</td>
      <td>-1.735934</td>
      <td>-2.696700</td>
      <td>2.455623</td>
      <td>-0.591208</td>
      <td>0.274002</td>
      <td>-5.769041</td>
      <td>-1.246138</td>
      <td>-0.044766</td>
      <td>-0.178684</td>
      <td>1.059918</td>
      <td>-1.444050</td>
      <td>-0.220837</td>
      <td>-0.434316</td>
      <td>-2.658855</td>
      <td>2.505102</td>
      <td>1.622691</td>
      <td>0.560247</td>
      <td>-0.232789</td>
      <td>-4.291800</td>
      <td>-0.568013</td>
      <td>2.583597</td>
      <td>0.486105</td>
      <td>1.730453</td>
      <td>-0.677343</td>
      <td>-0.684793</td>
      <td>-0.127586</td>
      <td>1.130949</td>
      <td>-0.471296</td>
      <td>0.406995</td>
      <td>-0.636077</td>
      <td>0.709149</td>
      <td>0.194866</td>
      <td>1.737456</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.306618</td>
      <td>-9.465927</td>
      <td>-4.280237</td>
      <td>0.747145</td>
      <td>-3.842896</td>
      <td>-1.208390</td>
      <td>-2.934347</td>
      <td>-2.197162</td>
      <td>-0.511378</td>
      <td>2.553976</td>
      <td>-0.510711</td>
      <td>1.405750</td>
      <td>-1.352031</td>
      <td>1.074685</td>
      <td>-0.928036</td>
      <td>1.833188</td>
      <td>2.295927</td>
      <td>-1.669374</td>
      <td>0.631459</td>
      <td>2.418053</td>
      <td>-0.078134</td>
      <td>-2.432385</td>
      <td>0.269018</td>
      <td>0.174332</td>
      <td>-1.215254</td>
      <td>-2.270264</td>
      <td>2.394836</td>
      <td>0.326568</td>
      <td>1.607412</td>
      <td>1.108830</td>
      <td>-1.255134</td>
      <td>-0.327092</td>
      <td>0.876635</td>
      <td>-2.208845</td>
      <td>-0.654325</td>
      <td>0.566157</td>
      <td>-1.348874</td>
      <td>-1.198655</td>
      <td>2.135779</td>
      <td>-0.575339</td>
      <td>-1.347962</td>
      <td>-0.427463</td>
      <td>-0.607911</td>
      <td>0.337198</td>
      <td>1.477528</td>
      <td>0.253536</td>
      <td>-0.247333</td>
      <td>-0.372204</td>
      <td>-0.907106</td>
      <td>-0.152948</td>
      <td>0.235667</td>
      <td>-0.025401</td>
      <td>0.749870</td>
      <td>0.541306</td>
      <td>-0.074447</td>
      <td>-2.240163</td>
      <td>-0.742984</td>
      <td>-2.546209</td>
      <td>0.703664</td>
      <td>-2.418568</td>
      <td>-2.549477</td>
      <td>2.344405</td>
      <td>0.475233</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### 2.4.1

We select the features of interest that may be relevant to identifying the genre of the song. In particular:
- acoustiness
- danceability
- energy
- instrumentalness
- liveness
- speechiness
- tempo
- valence
- duration


```python
## Extract from first dataset
d_pivot = df1[['audio_features_acousticness','audio_features_danceability','audio_features_energy', 'audio_features_instrumentalness', 'audio_features_liveness',
               'audio_features_speechiness', 'audio_features_tempo', 'audio_features_valence']]
```


```python
# Extract track duration
d_pivot['track_duration'] = df3.track_duration;
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    


```python
## Check
d_pivot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>audio_features_acousticness</th>
      <th>audio_features_danceability</th>
      <th>audio_features_energy</th>
      <th>audio_features_instrumentalness</th>
      <th>audio_features_liveness</th>
      <th>audio_features_speechiness</th>
      <th>audio_features_tempo</th>
      <th>audio_features_valence</th>
      <th>track_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.416675</td>
      <td>0.675894</td>
      <td>0.634476</td>
      <td>0.010628</td>
      <td>0.177647</td>
      <td>0.159310</td>
      <td>165.922</td>
      <td>0.576661</td>
      <td>168</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.374408</td>
      <td>0.528643</td>
      <td>0.817461</td>
      <td>0.001851</td>
      <td>0.105880</td>
      <td>0.461818</td>
      <td>126.957</td>
      <td>0.269240</td>
      <td>237</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.043567</td>
      <td>0.745566</td>
      <td>0.701470</td>
      <td>0.000697</td>
      <td>0.373143</td>
      <td>0.124595</td>
      <td>100.260</td>
      <td>0.621661</td>
      <td>206</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.951670</td>
      <td>0.658179</td>
      <td>0.924525</td>
      <td>0.965427</td>
      <td>0.115474</td>
      <td>0.032985</td>
      <td>111.562</td>
      <td>0.963590</td>
      <td>161</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.452217</td>
      <td>0.513238</td>
      <td>0.560410</td>
      <td>0.019443</td>
      <td>0.096567</td>
      <td>0.525519</td>
      <td>114.290</td>
      <td>0.894072</td>
      <td>311</td>
    </tr>
  </tbody>
</table>
</div>



### 2.4.2
If any of your selected variables are numerical (continuous or discrete), then categorize them into 4 categories.


```python
pd.options.mode.chained_assignment = None  
```


```python
## discrete each variable in 4 groups
for i in d_pivot.columns:
    d_pivot[i] = pd.qcut(d_pivot[i], q = 4)
```


```python
d_pivot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>audio_features_acousticness</th>
      <th>audio_features_danceability</th>
      <th>audio_features_energy</th>
      <th>audio_features_instrumentalness</th>
      <th>audio_features_liveness</th>
      <th>audio_features_speechiness</th>
      <th>audio_features_tempo</th>
      <th>audio_features_valence</th>
      <th>track_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(0.104, 0.574]</td>
      <td>(0.629, 0.969]</td>
      <td>(0.549, 0.776]</td>
      <td>(-0.001, 0.323]</td>
      <td>(0.119, 0.211]</td>
      <td>(0.0855, 0.966]</td>
      <td>(145.318, 251.072]</td>
      <td>(0.418, 0.666]</td>
      <td>(137.0, 200.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(0.104, 0.574]</td>
      <td>(0.486, 0.629]</td>
      <td>(0.776, 1.0]</td>
      <td>(-0.001, 0.323]</td>
      <td>(0.101, 0.119]</td>
      <td>(0.0855, 0.966]</td>
      <td>(120.057, 145.318]</td>
      <td>(0.197, 0.418]</td>
      <td>(200.0, 289.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(-0.000999096, 0.104]</td>
      <td>(0.629, 0.969]</td>
      <td>(0.549, 0.776]</td>
      <td>(-0.001, 0.323]</td>
      <td>(0.211, 0.98]</td>
      <td>(0.0855, 0.966]</td>
      <td>(95.967, 120.057]</td>
      <td>(0.418, 0.666]</td>
      <td>(200.0, 289.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(0.921, 0.996]</td>
      <td>(0.629, 0.969]</td>
      <td>(0.776, 1.0]</td>
      <td>(0.918, 0.998]</td>
      <td>(0.101, 0.119]</td>
      <td>(0.0213, 0.0369]</td>
      <td>(95.967, 120.057]</td>
      <td>(0.666, 1.0]</td>
      <td>(137.0, 200.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(0.104, 0.574]</td>
      <td>(0.486, 0.629]</td>
      <td>(0.549, 0.776]</td>
      <td>(-0.001, 0.323]</td>
      <td>(0.0243, 0.101]</td>
      <td>(0.0855, 0.966]</td>
      <td>(95.967, 120.057]</td>
      <td>(0.666, 1.0]</td>
      <td>(289.0, 18350.0]</td>
    </tr>
  </tbody>
</table>
</div>



### 2.4.3 - 2.4.4 - 2.4.5

Pivot table for Acoustincess.


```python
## Acoustincess
pv_a = pd.crosstab(d_pivot.audio_features_acousticness , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_a = pv_a.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_a.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_a = pv_a.append(pv_a.sum().rename('Total'))
```


```python
## Check
pv_a
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>13.69</td>
      <td>20.38</td>
      <td>0.16</td>
      <td>72.29</td>
      <td>18.62</td>
      <td>0.92</td>
      <td>11.29</td>
      <td>4.53</td>
      <td>35.45</td>
      <td>56.96</td>
      <td>2.13</td>
      <td>13.58</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>24.78</td>
      <td>34.98</td>
      <td>0.16</td>
      <td>23.40</td>
      <td>21.05</td>
      <td>3.66</td>
      <td>20.07</td>
      <td>12.76</td>
      <td>40.57</td>
      <td>28.68</td>
      <td>6.22</td>
      <td>27.16</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>39.49</td>
      <td>30.89</td>
      <td>3.15</td>
      <td>3.92</td>
      <td>30.36</td>
      <td>17.93</td>
      <td>31.14</td>
      <td>28.40</td>
      <td>20.91</td>
      <td>13.03</td>
      <td>27.33</td>
      <td>35.03</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>22.04</td>
      <td>13.75</td>
      <td>96.54</td>
      <td>0.39</td>
      <td>29.96</td>
      <td>77.49</td>
      <td>37.50</td>
      <td>54.32</td>
      <td>3.07</td>
      <td>1.33</td>
      <td>64.32</td>
      <td>24.22</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the fourth and tenth clusters, while the **second** category is prevalent in second and ninth cluster.
The **third** first, fifth and twelfth cluster.
The **fourth** category is prevalent in the third, sixth, seventh, eighth and eleventh clusters.

Pivot table for Danceability.


```python
## Danceability
pv_d = pd.crosstab(d_pivot.audio_features_danceability , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_d = pv_d.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_d.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_d = pv_d.append(pv_d.sum().rename('Total'))
```


```python
## Check
pv_d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>33.18</td>
      <td>7.61</td>
      <td>28.03</td>
      <td>4.71</td>
      <td>37.65</td>
      <td>38.22</td>
      <td>46.27</td>
      <td>42.80</td>
      <td>3.95</td>
      <td>36.87</td>
      <td>35.02</td>
      <td>22.23</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>30.77</td>
      <td>19.76</td>
      <td>39.84</td>
      <td>12.55</td>
      <td>24.70</td>
      <td>30.10</td>
      <td>25.00</td>
      <td>31.69</td>
      <td>10.16</td>
      <td>29.62</td>
      <td>30.28</td>
      <td>26.47</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>23.85</td>
      <td>32.80</td>
      <td>22.68</td>
      <td>23.79</td>
      <td>23.48</td>
      <td>24.35</td>
      <td>18.86</td>
      <td>16.87</td>
      <td>20.76</td>
      <td>23.79</td>
      <td>25.70</td>
      <td>28.11</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>12.20</td>
      <td>39.83</td>
      <td>9.45</td>
      <td>58.95</td>
      <td>14.17</td>
      <td>7.33</td>
      <td>9.87</td>
      <td>8.64</td>
      <td>65.13</td>
      <td>9.72</td>
      <td>9.00</td>
      <td>23.18</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the first, fifth, sixth, seventh, eighth, tenth and eleventh clusters, while the **second** category is prevalent in the third cluster.
The **third** category is prevalent in the twelfth cluster.
The **fourth** category is prevalent in the second, fourth and ninth clusters.

Pivot table for Energy.


```python
## Energy
pv_e = pd.crosstab(d_pivot.audio_features_energy , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_e = pv_e.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_e.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_e = pv_e.append(pv_e.sum().rename('Total'))
```


```python
## Check
pv_e
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>4.69</td>
      <td>17.22</td>
      <td>86.77</td>
      <td>5.62</td>
      <td>37.25</td>
      <td>81.15</td>
      <td>10.64</td>
      <td>65.02</td>
      <td>7.16</td>
      <td>2.17</td>
      <td>79.87</td>
      <td>52.25</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>22.92</td>
      <td>40.59</td>
      <td>12.28</td>
      <td>28.37</td>
      <td>33.60</td>
      <td>16.75</td>
      <td>21.82</td>
      <td>22.22</td>
      <td>27.56</td>
      <td>11.06</td>
      <td>16.37</td>
      <td>35.99</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>37.96</td>
      <td>30.62</td>
      <td>0.94</td>
      <td>33.59</td>
      <td>16.19</td>
      <td>1.96</td>
      <td>28.07</td>
      <td>10.29</td>
      <td>35.16</td>
      <td>27.34</td>
      <td>3.27</td>
      <td>10.55</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>34.43</td>
      <td>11.57</td>
      <td>0.00</td>
      <td>32.42</td>
      <td>12.96</td>
      <td>0.13</td>
      <td>39.47</td>
      <td>2.47</td>
      <td>30.12</td>
      <td>59.43</td>
      <td>0.49</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the third, fifth, sixth, eighth, eleventh and twelfth clusters, while the **second** category is  prevalent in second cluster.
The **third** category is prevalent in the first, fourth and ninthclusters.
The **fourth** category is prevalent in the seventh and tenth clusters.

Pivot table for Instrumentalness.


```python
## Instrumentalness
pv_i = pd.crosstab(d_pivot.audio_features_instrumentalness , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_i = pv_i.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_i.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_i = pv_i.append(pv_i.sum().rename('Total'))
```


```python
## Check
pv_i
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>23.11</td>
      <td>43.08</td>
      <td>19.21</td>
      <td>14.77</td>
      <td>7.69</td>
      <td>25.65</td>
      <td>10.64</td>
      <td>7.82</td>
      <td>27.12</td>
      <td>23.89</td>
      <td>22.26</td>
      <td>22.49</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>29.74</td>
      <td>26.21</td>
      <td>10.87</td>
      <td>30.20</td>
      <td>18.22</td>
      <td>20.68</td>
      <td>18.86</td>
      <td>16.46</td>
      <td>24.49</td>
      <td>29.71</td>
      <td>21.44</td>
      <td>23.27</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>25.06</td>
      <td>18.42</td>
      <td>13.70</td>
      <td>38.30</td>
      <td>24.29</td>
      <td>20.94</td>
      <td>24.56</td>
      <td>26.75</td>
      <td>32.89</td>
      <td>25.52</td>
      <td>24.06</td>
      <td>28.11</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>22.09</td>
      <td>12.28</td>
      <td>56.22</td>
      <td>16.73</td>
      <td>49.80</td>
      <td>32.72</td>
      <td>45.94</td>
      <td>48.97</td>
      <td>15.50</td>
      <td>20.88</td>
      <td>32.24</td>
      <td>26.12</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the second cluster, while the **second** category is prevalent first and tenth cluster.
The **third** category is prevalent in the fourth, ninth and twelfth clusters.
The **fourth** category is prevalent in the third, fifth, sixth, seventh, eighth and twelfth clusters.

Pivot table for Liveness.


```python
## Liveness
pv_l = pd.crosstab(d_pivot.audio_features_liveness , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_l = pv_l.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_l.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_l = pv_l.append(pv_l.sum().rename('Total'))
```


```python
## Check
pv_l
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>19.49</td>
      <td>29.77</td>
      <td>9.61</td>
      <td>33.73</td>
      <td>23.08</td>
      <td>22.91</td>
      <td>16.23</td>
      <td>25.51</td>
      <td>33.55</td>
      <td>26.46</td>
      <td>25.70</td>
      <td>24.31</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>21.90</td>
      <td>24.97</td>
      <td>15.12</td>
      <td>22.22</td>
      <td>32.39</td>
      <td>36.52</td>
      <td>23.03</td>
      <td>40.74</td>
      <td>22.44</td>
      <td>18.31</td>
      <td>36.17</td>
      <td>35.99</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>27.70</td>
      <td>25.23</td>
      <td>23.31</td>
      <td>20.65</td>
      <td>29.96</td>
      <td>22.12</td>
      <td>31.47</td>
      <td>22.63</td>
      <td>20.10</td>
      <td>27.54</td>
      <td>22.26</td>
      <td>22.32</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>30.90</td>
      <td>20.03</td>
      <td>51.97</td>
      <td>23.40</td>
      <td>14.57</td>
      <td>18.46</td>
      <td>29.28</td>
      <td>11.11</td>
      <td>23.90</td>
      <td>27.69</td>
      <td>15.88</td>
      <td>17.39</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.01</td>
      <td>99.99</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.01</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the second, fourth and ninth clusters, while the **second** category is prevalent in the second, seventh, ninth, eleventh and twelfth clusters.
The **third** category is prevalent in the seventh cluster.
The **fourth** category is prevalent in the first third and tenth clusters.

Pivot table for Speechiness.


```python
## Speechiness
pv_s = pd.crosstab(d_pivot.audio_features_speechiness , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_s = pv_s.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_s.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_s = pv_s.append(pv_s.sum().rename('Total'))
```


```python
## Check
pv_s
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>25.24</td>
      <td>26.75</td>
      <td>19.69</td>
      <td>6.80</td>
      <td>39.68</td>
      <td>39.53</td>
      <td>26.10</td>
      <td>40.74</td>
      <td>1.83</td>
      <td>28.33</td>
      <td>40.59</td>
      <td>32.61</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>28.35</td>
      <td>20.65</td>
      <td>31.02</td>
      <td>18.43</td>
      <td>29.15</td>
      <td>37.96</td>
      <td>27.96</td>
      <td>36.63</td>
      <td>8.04</td>
      <td>25.37</td>
      <td>37.32</td>
      <td>26.90</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>26.68</td>
      <td>20.74</td>
      <td>23.94</td>
      <td>36.08</td>
      <td>23.89</td>
      <td>16.10</td>
      <td>29.61</td>
      <td>18.52</td>
      <td>26.10</td>
      <td>30.21</td>
      <td>16.86</td>
      <td>21.11</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>19.72</td>
      <td>31.86</td>
      <td>25.35</td>
      <td>38.69</td>
      <td>7.29</td>
      <td>6.41</td>
      <td>16.34</td>
      <td>4.12</td>
      <td>64.04</td>
      <td>16.09</td>
      <td>5.24</td>
      <td>19.38</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.01</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the fifth, sixth, eighth, eleventh and twelfth clusters, while the **second** category is prevalent in the first and third clusters.
The **third** category is prevalent in the seventh and tenth clusters.
The **fourth** category is prevalent in the second, fourth and ninth clusters.

Pivot table for Tempo.


```python
## Tempo
pv_t = pd.crosstab(d_pivot.audio_features_tempo , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_t = pv_t.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_t.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_t = pv_t.append(pv_t.sum().rename('Total'))
```


```python
## Check
pv_t
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>17.77</td>
      <td>26.17</td>
      <td>35.12</td>
      <td>24.05</td>
      <td>23.48</td>
      <td>37.96</td>
      <td>22.26</td>
      <td>30.86</td>
      <td>25.66</td>
      <td>16.98</td>
      <td>37.32</td>
      <td>30.80</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>22.74</td>
      <td>26.26</td>
      <td>26.30</td>
      <td>23.40</td>
      <td>29.96</td>
      <td>28.27</td>
      <td>24.01</td>
      <td>24.28</td>
      <td>24.49</td>
      <td>22.36</td>
      <td>26.68</td>
      <td>29.15</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>27.19</td>
      <td>23.19</td>
      <td>20.63</td>
      <td>27.58</td>
      <td>26.72</td>
      <td>20.68</td>
      <td>23.36</td>
      <td>23.46</td>
      <td>25.95</td>
      <td>29.62</td>
      <td>20.29</td>
      <td>22.49</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>32.30</td>
      <td>24.39</td>
      <td>17.95</td>
      <td>24.97</td>
      <td>19.84</td>
      <td>13.09</td>
      <td>30.37</td>
      <td>21.40</td>
      <td>23.90</td>
      <td>31.05</td>
      <td>15.71</td>
      <td>17.56</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the third, sixth, eighth, eleven and twelfth clusters while the **second** category is prevalent in the second and fifth clusters.
The **third** category is prevalent in the fourth cluster.
The **fourth** category is prevalent in the first, seventh tenth clusters.

Pivot table for Valence.


```python
## Valence
pv_v = pd.crosstab(d_pivot.audio_features_valence , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_v = pv_v.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_v.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_v = pv_v.append(pv_v.sum().rename('Total'))
```


```python
## Check
pv_v
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>17.59</td>
      <td>15.22</td>
      <td>19.06</td>
      <td>20.78</td>
      <td>49.80</td>
      <td>44.63</td>
      <td>34.54</td>
      <td>55.97</td>
      <td>13.67</td>
      <td>17.13</td>
      <td>50.57</td>
      <td>45.33</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>25.10</td>
      <td>22.92</td>
      <td>29.92</td>
      <td>26.67</td>
      <td>22.27</td>
      <td>29.58</td>
      <td>22.15</td>
      <td>18.52</td>
      <td>17.69</td>
      <td>28.04</td>
      <td>28.31</td>
      <td>27.77</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>25.75</td>
      <td>30.17</td>
      <td>24.41</td>
      <td>28.76</td>
      <td>18.62</td>
      <td>16.75</td>
      <td>22.15</td>
      <td>16.05</td>
      <td>29.82</td>
      <td>28.18</td>
      <td>11.46</td>
      <td>18.17</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>31.55</td>
      <td>31.69</td>
      <td>26.61</td>
      <td>23.79</td>
      <td>9.31</td>
      <td>9.03</td>
      <td>21.16</td>
      <td>9.47</td>
      <td>38.82</td>
      <td>26.65</td>
      <td>9.66</td>
      <td>8.74</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the fifth, sixth, seventh, eighth, eleventh and twelfth clusters while the **second** category is prevalent in the third cluster.
The **third** category is prevalent in the fourth and tenth clusters.
The **fourth** category is prevalent in the first, second and ninth  clusters.

Pivot table for Duration.


```python
## Duration
pv_d = pd.crosstab(d_pivot.track_duration , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_d = pv_d.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Index Row
pv_d.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_d = pv_d.append(pv_d.sum().rename('Total'))
```


```python
## Check
pv_d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>24.97</td>
      <td>25.86</td>
      <td>29.45</td>
      <td>24.71</td>
      <td>29.55</td>
      <td>28.27</td>
      <td>24.12</td>
      <td>20.99</td>
      <td>25.95</td>
      <td>24.19</td>
      <td>24.06</td>
      <td>23.62</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>26.13</td>
      <td>24.92</td>
      <td>23.46</td>
      <td>22.75</td>
      <td>23.08</td>
      <td>24.48</td>
      <td>23.14</td>
      <td>25.93</td>
      <td>26.61</td>
      <td>25.02</td>
      <td>26.35</td>
      <td>24.48</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>23.06</td>
      <td>25.10</td>
      <td>21.73</td>
      <td>27.58</td>
      <td>25.51</td>
      <td>24.08</td>
      <td>25.00</td>
      <td>26.75</td>
      <td>25.00</td>
      <td>24.78</td>
      <td>25.37</td>
      <td>27.77</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>25.85</td>
      <td>24.12</td>
      <td>25.35</td>
      <td>24.97</td>
      <td>21.86</td>
      <td>23.17</td>
      <td>27.74</td>
      <td>26.34</td>
      <td>22.44</td>
      <td>26.01</td>
      <td>24.22</td>
      <td>24.13</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.01</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



The **first** category is prevalent in the second, third, fifth and ninth clusters while the **second** category is prevalent in the first, ninth and eleventh clusters.
The **third** category is prevalent in the fourth, eighth and twelfth clusters.
The **fourth** category is prevalent in the seventh and tenth clusters.

### 2.4.6
What is the most representative genre for each one of the clusters?


```python
## Sup Join
sup = pd.merge(left = df1, right = df3, left_on = 'track_id', right_on = 'track_id')
## Extract Genre
song_genre = sup.track_genre_top
```

Compare the obtained clusters to the reality genre.


```python
## Genre
pv_g = pd.crosstab(song_genre , cluster_12,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_g = pv_g.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8', 8: 'Cluster 9', 9: 'Cluster 10', 10: 'Cluster 11', 11: 'Cluster 12',}, axis=1)
## Add Total
pv_g = pv_g.append(pv_g.sum().rename('Total'))
```


```python
## Check
pv_g
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
      <th>Cluster 9</th>
      <th>Cluster 10</th>
      <th>Cluster 11</th>
      <th>Cluster 12</th>
    </tr>
    <tr>
      <th>track_genre_top</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Blues</th>
      <td>0.88</td>
      <td>0.91</td>
      <td>0.59</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.36</td>
      <td>0.14</td>
      <td>0.00</td>
      <td>0.52</td>
      <td>0.00</td>
      <td>1.45</td>
      <td>1.41</td>
    </tr>
    <tr>
      <th>Classical</th>
      <td>0.29</td>
      <td>0.23</td>
      <td>8.89</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>19.78</td>
      <td>0.95</td>
      <td>4.40</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>18.07</td>
      <td>1.67</td>
    </tr>
    <tr>
      <th>Electronic</th>
      <td>7.75</td>
      <td>30.18</td>
      <td>1.58</td>
      <td>74.91</td>
      <td>23.76</td>
      <td>8.35</td>
      <td>13.37</td>
      <td>23.27</td>
      <td>55.21</td>
      <td>15.45</td>
      <td>13.73</td>
      <td>24.23</td>
    </tr>
    <tr>
      <th>Experimental</th>
      <td>0.06</td>
      <td>0.30</td>
      <td>0.79</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.18</td>
      <td>0.14</td>
      <td>0.00</td>
      <td>0.10</td>
      <td>0.26</td>
      <td>0.24</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Folk</th>
      <td>3.70</td>
      <td>7.26</td>
      <td>13.44</td>
      <td>1.70</td>
      <td>18.23</td>
      <td>37.02</td>
      <td>10.78</td>
      <td>37.11</td>
      <td>0.52</td>
      <td>2.18</td>
      <td>34.94</td>
      <td>10.26</td>
    </tr>
    <tr>
      <th>Hip-Hop</th>
      <td>6.58</td>
      <td>20.50</td>
      <td>0.40</td>
      <td>16.04</td>
      <td>3.87</td>
      <td>0.91</td>
      <td>5.87</td>
      <td>1.26</td>
      <td>30.42</td>
      <td>4.16</td>
      <td>0.48</td>
      <td>3.33</td>
    </tr>
    <tr>
      <th>Instrumental</th>
      <td>0.53</td>
      <td>1.13</td>
      <td>0.40</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.73</td>
      <td>0.95</td>
      <td>1.89</td>
      <td>0.31</td>
      <td>0.46</td>
      <td>2.17</td>
      <td>3.21</td>
    </tr>
    <tr>
      <th>International</th>
      <td>1.17</td>
      <td>2.42</td>
      <td>1.78</td>
      <td>0.00</td>
      <td>6.08</td>
      <td>2.18</td>
      <td>2.73</td>
      <td>1.26</td>
      <td>0.10</td>
      <td>0.20</td>
      <td>4.34</td>
      <td>0.64</td>
    </tr>
    <tr>
      <th>Jazz</th>
      <td>1.59</td>
      <td>2.72</td>
      <td>2.77</td>
      <td>0.38</td>
      <td>0.55</td>
      <td>3.09</td>
      <td>1.23</td>
      <td>3.14</td>
      <td>0.83</td>
      <td>0.73</td>
      <td>6.75</td>
      <td>10.64</td>
    </tr>
    <tr>
      <th>Old-Time / Historic</th>
      <td>0.23</td>
      <td>0.08</td>
      <td>63.83</td>
      <td>0.00</td>
      <td>1.10</td>
      <td>2.36</td>
      <td>0.82</td>
      <td>2.52</td>
      <td>0.10</td>
      <td>0.00</td>
      <td>0.72</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Pop</th>
      <td>2.94</td>
      <td>8.09</td>
      <td>1.38</td>
      <td>1.13</td>
      <td>2.21</td>
      <td>4.17</td>
      <td>2.59</td>
      <td>2.52</td>
      <td>1.98</td>
      <td>3.30</td>
      <td>2.65</td>
      <td>5.90</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>74.28</td>
      <td>26.17</td>
      <td>4.15</td>
      <td>5.85</td>
      <td>43.65</td>
      <td>18.87</td>
      <td>60.44</td>
      <td>22.64</td>
      <td>9.90</td>
      <td>73.27</td>
      <td>14.46</td>
      <td>38.72</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.01</td>
      <td>99.99</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.01</td>
    </tr>
  </tbody>
</table>
</div>



* **Blues, Experimental, Instrumental, International**: these genres contribute to a very small extent to the various clusters. This means that in those respective clusters where the contribution is almost null, there are few songs related to these genres.
By shifting the focus to clusters to understand which genre contributes most to that group, we note that:

* **Cluster 1**: is mainly characterized by *Rock*, with a percentage of 74.28%, which means that the first cluster identifies 74.28% of its observations of songs belonging to the genre *Rock*. An important but much smaller contribution is made by the genre *Electronic* with a value of 7.75% and the genre *Hip-Hop* with a value of 6.58%.

* **Cluster 2**: compared to the previous cluster, it is interesting to note that the percentage related to the contribution of the genre *Electronic* increases significantly, becoming 30.18%, as well as significantly increases the percentage of the genre *Hip-Hop* becoming 20.50%; decreases instead the value that refers to the genre *Rock*. Interesting to note is the increase in the percentage contribution of genres such as *Pop* and *Folk*.

* **Cluster 3**: compared to the previous cluster significantly increases the percentage contribution of the genre *Old-Time/Historic* which becomes dominant with a value of 63.83%, thus becoming dominant within this cluster. An important contribution is also made by the genre *Folk* with a value of 13.44%.

* **Cluster 4**: is undoubtedly characterized by genre songs *Electronic* with a percentage contribution of 74.91%. 

* **Cluster 5**: is characterized by *Rock* songs with a percentage contribution of 43.65%,  *Electronic* songs for 23.76% and *Folk* songs for 18.23%.

* **Cluster 6**: the dominant genre is *Folk* with a contribution of 37.02%, followed by the genre *Classical* with 19.78% of the contribution and the genre *Rock* with 18.87% of the contribution.

* **Cluster 7**: we still find the *Rock* as dominant genre with a percentage contribution of 60.44%, followed by the genre *Electronic and *Folk* with a percentage contribution of 13.37% and 10.78%.

* **Cluster 8**: the dominant genre is *Folk* with a percentage contribution of 37.11%, followed by the genre *Electronic* with a percentage contribution of 23.27%. 

* **Cluster 9**: characterized mainly by the genre *Electronic* with a contribution percentage of 55.21% and followed by the genre *Hip-Hop* with a contribution percentage of 30.42%.

* **Cluster 10**: the dominant genre is *Rock* with a percentage contribution of 73.27%.

* **Cluster 11**: the dominant genre is *Folk* with a percentage contribution of 34.94%, followed by the genre *Classical* with 18.07%.

* **Cluster 12**: the dominant genre is *Rock* with a percentage contribution of 38.72%, followed by the genre *Electronic* with 24.23%.

> Conclusion:

It is clear that the unsupervised learning approach is quite complex due to the impossibility of comparing the results obtained with the true class of reference.
In real problems in which *Clustering* techniques are used, it is very important to have professional experts in the domain who know the nature of the problem under consideration and can actually suggest what are the features that have an important impact on the various clusters and suggest based on what features the cluster was made.

In this case, for example, data from technical specifications relating to songs were processed; it would have been useful, for example, to collaborate with a professional figure who understood the technical meaning of the features.
This means that if, for example, the *j* cluster identifies *Rock* and *Pop* as the prevailing genre, there may be some characteristic that unites these two tracks, such as tempo, melody, tonality, timbre, etc.

### 2.4.7


```python
## we do not use 'echonest.csv' variables.
df2 = pd.read_csv('features.csv')
```


```python
## Info
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 106574 entries, 0 to 106573
    Columns: 519 entries, track_id to zcr_std_01
    dtypes: float64(518), int64(1)
    memory usage: 422.0 MB
    


```python
## Check
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track_id</th>
      <th>chroma_cens_kurtosis_01</th>
      <th>chroma_cens_kurtosis_02</th>
      <th>chroma_cens_kurtosis_03</th>
      <th>chroma_cens_kurtosis_04</th>
      <th>chroma_cens_kurtosis_05</th>
      <th>chroma_cens_kurtosis_06</th>
      <th>chroma_cens_kurtosis_07</th>
      <th>chroma_cens_kurtosis_08</th>
      <th>chroma_cens_kurtosis_09</th>
      <th>chroma_cens_kurtosis_10</th>
      <th>chroma_cens_kurtosis_11</th>
      <th>chroma_cens_kurtosis_12</th>
      <th>chroma_cens_max_01</th>
      <th>chroma_cens_max_02</th>
      <th>chroma_cens_max_03</th>
      <th>chroma_cens_max_04</th>
      <th>chroma_cens_max_05</th>
      <th>chroma_cens_max_06</th>
      <th>chroma_cens_max_07</th>
      <th>chroma_cens_max_08</th>
      <th>chroma_cens_max_09</th>
      <th>chroma_cens_max_10</th>
      <th>chroma_cens_max_11</th>
      <th>chroma_cens_max_12</th>
      <th>chroma_cens_mean_01</th>
      <th>chroma_cens_mean_02</th>
      <th>chroma_cens_mean_03</th>
      <th>chroma_cens_mean_04</th>
      <th>chroma_cens_mean_05</th>
      <th>chroma_cens_mean_06</th>
      <th>chroma_cens_mean_07</th>
      <th>chroma_cens_mean_08</th>
      <th>chroma_cens_mean_09</th>
      <th>chroma_cens_mean_10</th>
      <th>chroma_cens_mean_11</th>
      <th>chroma_cens_mean_12</th>
      <th>chroma_cens_median_01</th>
      <th>chroma_cens_median_02</th>
      <th>chroma_cens_median_03</th>
      <th>...</th>
      <th>tonnetz_max_04</th>
      <th>tonnetz_max_05</th>
      <th>tonnetz_max_06</th>
      <th>tonnetz_mean_01</th>
      <th>tonnetz_mean_02</th>
      <th>tonnetz_mean_03</th>
      <th>tonnetz_mean_04</th>
      <th>tonnetz_mean_05</th>
      <th>tonnetz_mean_06</th>
      <th>tonnetz_median_01</th>
      <th>tonnetz_median_02</th>
      <th>tonnetz_median_03</th>
      <th>tonnetz_median_04</th>
      <th>tonnetz_median_05</th>
      <th>tonnetz_median_06</th>
      <th>tonnetz_min_01</th>
      <th>tonnetz_min_02</th>
      <th>tonnetz_min_03</th>
      <th>tonnetz_min_04</th>
      <th>tonnetz_min_05</th>
      <th>tonnetz_min_06</th>
      <th>tonnetz_skew_01</th>
      <th>tonnetz_skew_02</th>
      <th>tonnetz_skew_03</th>
      <th>tonnetz_skew_04</th>
      <th>tonnetz_skew_05</th>
      <th>tonnetz_skew_06</th>
      <th>tonnetz_std_01</th>
      <th>tonnetz_std_02</th>
      <th>tonnetz_std_03</th>
      <th>tonnetz_std_04</th>
      <th>tonnetz_std_05</th>
      <th>tonnetz_std_06</th>
      <th>zcr_kurtosis_01</th>
      <th>zcr_max_01</th>
      <th>zcr_mean_01</th>
      <th>zcr_median_01</th>
      <th>zcr_min_01</th>
      <th>zcr_skew_01</th>
      <th>zcr_std_01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>7.180653</td>
      <td>5.230309</td>
      <td>0.249321</td>
      <td>1.347620</td>
      <td>1.482478</td>
      <td>0.531371</td>
      <td>1.481593</td>
      <td>2.691455</td>
      <td>0.866868</td>
      <td>1.341231</td>
      <td>1.347792</td>
      <td>1.237658</td>
      <td>0.692500</td>
      <td>0.569344</td>
      <td>0.597041</td>
      <td>0.625864</td>
      <td>0.567330</td>
      <td>0.443949</td>
      <td>0.487976</td>
      <td>0.497327</td>
      <td>0.574435</td>
      <td>0.579241</td>
      <td>0.620102</td>
      <td>0.586945</td>
      <td>0.474300</td>
      <td>0.369816</td>
      <td>0.236119</td>
      <td>0.228068</td>
      <td>0.222830</td>
      <td>0.221415</td>
      <td>0.229238</td>
      <td>0.248795</td>
      <td>0.196245</td>
      <td>0.175809</td>
      <td>0.200713</td>
      <td>0.319972</td>
      <td>0.482825</td>
      <td>0.387652</td>
      <td>0.249082</td>
      <td>...</td>
      <td>0.318972</td>
      <td>0.059690</td>
      <td>0.069184</td>
      <td>-0.002570</td>
      <td>0.019296</td>
      <td>0.010510</td>
      <td>0.073464</td>
      <td>0.009272</td>
      <td>0.015765</td>
      <td>-0.003789</td>
      <td>0.017786</td>
      <td>0.007311</td>
      <td>0.067945</td>
      <td>0.009488</td>
      <td>0.016876</td>
      <td>-0.059769</td>
      <td>-0.091745</td>
      <td>-0.185687</td>
      <td>-0.140306</td>
      <td>-0.048525</td>
      <td>-0.089286</td>
      <td>0.752462</td>
      <td>0.262607</td>
      <td>0.200944</td>
      <td>0.593595</td>
      <td>-0.177665</td>
      <td>-1.424201</td>
      <td>0.019809</td>
      <td>0.029569</td>
      <td>0.038974</td>
      <td>0.054125</td>
      <td>0.012226</td>
      <td>0.012111</td>
      <td>5.758890</td>
      <td>0.459473</td>
      <td>0.085629</td>
      <td>0.071289</td>
      <td>0.000000</td>
      <td>2.089872</td>
      <td>0.061448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1.888963</td>
      <td>0.760539</td>
      <td>0.345297</td>
      <td>2.295201</td>
      <td>1.654031</td>
      <td>0.067592</td>
      <td>1.366848</td>
      <td>1.054094</td>
      <td>0.108103</td>
      <td>0.619185</td>
      <td>1.038253</td>
      <td>1.292235</td>
      <td>0.677641</td>
      <td>0.584248</td>
      <td>0.581271</td>
      <td>0.581182</td>
      <td>0.454241</td>
      <td>0.464841</td>
      <td>0.542833</td>
      <td>0.664720</td>
      <td>0.511329</td>
      <td>0.530998</td>
      <td>0.603398</td>
      <td>0.547428</td>
      <td>0.232784</td>
      <td>0.229469</td>
      <td>0.225674</td>
      <td>0.216713</td>
      <td>0.220512</td>
      <td>0.242744</td>
      <td>0.369235</td>
      <td>0.420716</td>
      <td>0.312129</td>
      <td>0.242748</td>
      <td>0.264292</td>
      <td>0.225683</td>
      <td>0.230579</td>
      <td>0.228059</td>
      <td>0.209370</td>
      <td>...</td>
      <td>0.214807</td>
      <td>0.070261</td>
      <td>0.070394</td>
      <td>0.000183</td>
      <td>0.006908</td>
      <td>0.047025</td>
      <td>-0.029942</td>
      <td>0.017535</td>
      <td>-0.001496</td>
      <td>-0.000108</td>
      <td>0.007161</td>
      <td>0.046912</td>
      <td>-0.021149</td>
      <td>0.016299</td>
      <td>-0.002657</td>
      <td>-0.097199</td>
      <td>-0.079651</td>
      <td>-0.164613</td>
      <td>-0.304375</td>
      <td>-0.024958</td>
      <td>-0.055667</td>
      <td>0.265541</td>
      <td>-0.131471</td>
      <td>0.171930</td>
      <td>-0.990710</td>
      <td>0.574556</td>
      <td>0.556494</td>
      <td>0.026316</td>
      <td>0.018708</td>
      <td>0.051151</td>
      <td>0.063831</td>
      <td>0.014212</td>
      <td>0.017740</td>
      <td>2.824694</td>
      <td>0.466309</td>
      <td>0.084578</td>
      <td>0.063965</td>
      <td>0.000000</td>
      <td>1.716724</td>
      <td>0.069330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0.527563</td>
      <td>-0.077654</td>
      <td>-0.279610</td>
      <td>0.685883</td>
      <td>1.937570</td>
      <td>0.880839</td>
      <td>-0.923192</td>
      <td>-0.927232</td>
      <td>0.666617</td>
      <td>1.038546</td>
      <td>0.268932</td>
      <td>1.125141</td>
      <td>0.611014</td>
      <td>0.651471</td>
      <td>0.494528</td>
      <td>0.448799</td>
      <td>0.468624</td>
      <td>0.454021</td>
      <td>0.497172</td>
      <td>0.559755</td>
      <td>0.671287</td>
      <td>0.610565</td>
      <td>0.551663</td>
      <td>0.603413</td>
      <td>0.258420</td>
      <td>0.303385</td>
      <td>0.250737</td>
      <td>0.218562</td>
      <td>0.245743</td>
      <td>0.236018</td>
      <td>0.275766</td>
      <td>0.293982</td>
      <td>0.346324</td>
      <td>0.289821</td>
      <td>0.246368</td>
      <td>0.220939</td>
      <td>0.255472</td>
      <td>0.293571</td>
      <td>0.245253</td>
      <td>...</td>
      <td>0.180027</td>
      <td>0.072169</td>
      <td>0.076847</td>
      <td>-0.007501</td>
      <td>-0.018525</td>
      <td>-0.030318</td>
      <td>0.024743</td>
      <td>0.004771</td>
      <td>-0.004536</td>
      <td>-0.007385</td>
      <td>-0.018953</td>
      <td>-0.020358</td>
      <td>0.024615</td>
      <td>0.004868</td>
      <td>-0.003899</td>
      <td>-0.128391</td>
      <td>-0.125289</td>
      <td>-0.359463</td>
      <td>-0.166667</td>
      <td>-0.038546</td>
      <td>-0.146136</td>
      <td>1.212025</td>
      <td>0.218381</td>
      <td>-0.419971</td>
      <td>-0.014541</td>
      <td>-0.199314</td>
      <td>-0.925733</td>
      <td>0.025550</td>
      <td>0.021106</td>
      <td>0.084997</td>
      <td>0.040730</td>
      <td>0.012691</td>
      <td>0.014759</td>
      <td>6.808415</td>
      <td>0.375000</td>
      <td>0.053114</td>
      <td>0.041504</td>
      <td>0.000000</td>
      <td>2.193303</td>
      <td>0.044861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>3.702245</td>
      <td>-0.291193</td>
      <td>2.196742</td>
      <td>-0.234449</td>
      <td>1.367364</td>
      <td>0.998411</td>
      <td>1.770694</td>
      <td>1.604566</td>
      <td>0.521217</td>
      <td>1.982386</td>
      <td>4.326824</td>
      <td>1.300406</td>
      <td>0.461840</td>
      <td>0.540411</td>
      <td>0.446708</td>
      <td>0.647553</td>
      <td>0.591908</td>
      <td>0.513306</td>
      <td>0.651501</td>
      <td>0.516887</td>
      <td>0.511479</td>
      <td>0.478263</td>
      <td>0.638766</td>
      <td>0.638495</td>
      <td>0.229882</td>
      <td>0.286978</td>
      <td>0.240096</td>
      <td>0.226792</td>
      <td>0.192443</td>
      <td>0.288410</td>
      <td>0.413348</td>
      <td>0.349137</td>
      <td>0.268424</td>
      <td>0.243144</td>
      <td>0.268941</td>
      <td>0.236763</td>
      <td>0.230555</td>
      <td>0.280229</td>
      <td>0.234060</td>
      <td>...</td>
      <td>0.192640</td>
      <td>0.117094</td>
      <td>0.059757</td>
      <td>-0.021650</td>
      <td>-0.018369</td>
      <td>-0.003282</td>
      <td>-0.074165</td>
      <td>0.008971</td>
      <td>0.007101</td>
      <td>-0.021108</td>
      <td>-0.019117</td>
      <td>-0.007409</td>
      <td>-0.067350</td>
      <td>0.007036</td>
      <td>0.006788</td>
      <td>-0.107889</td>
      <td>-0.194957</td>
      <td>-0.273549</td>
      <td>-0.343055</td>
      <td>-0.052284</td>
      <td>-0.029836</td>
      <td>-0.135219</td>
      <td>-0.275780</td>
      <td>0.015767</td>
      <td>-1.094873</td>
      <td>1.164041</td>
      <td>0.246746</td>
      <td>0.021413</td>
      <td>0.031989</td>
      <td>0.088197</td>
      <td>0.074358</td>
      <td>0.017952</td>
      <td>0.013921</td>
      <td>21.434212</td>
      <td>0.452148</td>
      <td>0.077515</td>
      <td>0.071777</td>
      <td>0.000000</td>
      <td>3.542325</td>
      <td>0.040800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>-0.193837</td>
      <td>-0.198527</td>
      <td>0.201546</td>
      <td>0.258556</td>
      <td>0.775204</td>
      <td>0.084794</td>
      <td>-0.289294</td>
      <td>-0.816410</td>
      <td>0.043851</td>
      <td>-0.804761</td>
      <td>-0.990958</td>
      <td>-0.430381</td>
      <td>0.652864</td>
      <td>0.676290</td>
      <td>0.670288</td>
      <td>0.598666</td>
      <td>0.653607</td>
      <td>0.697645</td>
      <td>0.664929</td>
      <td>0.686563</td>
      <td>0.635117</td>
      <td>0.667393</td>
      <td>0.689589</td>
      <td>0.683196</td>
      <td>0.202806</td>
      <td>0.245125</td>
      <td>0.262997</td>
      <td>0.187961</td>
      <td>0.182397</td>
      <td>0.238173</td>
      <td>0.278600</td>
      <td>0.292905</td>
      <td>0.247150</td>
      <td>0.292501</td>
      <td>0.304655</td>
      <td>0.235177</td>
      <td>0.200830</td>
      <td>0.224745</td>
      <td>0.234279</td>
      <td>...</td>
      <td>0.286794</td>
      <td>0.097534</td>
      <td>0.072202</td>
      <td>0.012362</td>
      <td>0.012246</td>
      <td>-0.021837</td>
      <td>-0.075866</td>
      <td>0.006179</td>
      <td>-0.007771</td>
      <td>0.011057</td>
      <td>0.012416</td>
      <td>-0.025059</td>
      <td>-0.072732</td>
      <td>0.005057</td>
      <td>-0.006812</td>
      <td>-0.147339</td>
      <td>-0.210110</td>
      <td>-0.342446</td>
      <td>-0.388083</td>
      <td>-0.075566</td>
      <td>-0.091831</td>
      <td>0.192395</td>
      <td>-0.215337</td>
      <td>0.081732</td>
      <td>0.040777</td>
      <td>0.232350</td>
      <td>-0.207831</td>
      <td>0.033342</td>
      <td>0.035174</td>
      <td>0.105521</td>
      <td>0.095003</td>
      <td>0.022492</td>
      <td>0.021355</td>
      <td>16.669037</td>
      <td>0.469727</td>
      <td>0.047225</td>
      <td>0.040039</td>
      <td>0.000977</td>
      <td>3.189831</td>
      <td>0.030993</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 519 columns</p>
</div>




```python
## Drop track id
df2 = df2.drop(['track_id'], axis=1)
```


```python
## Info
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 106574 entries, 0 to 106573
    Columns: 518 entries, chroma_cens_kurtosis_01 to zcr_std_01
    dtypes: float64(518)
    memory usage: 421.2 MB
    

Dimensionality Reduction


```python
## Standardization
scaler = StandardScaler()
df_scale2 = scaler.fit_transform(df2)
df_scale2 = pd.DataFrame(df_scale2)
df_scale2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>...</th>
      <th>478</th>
      <th>479</th>
      <th>480</th>
      <th>481</th>
      <th>482</th>
      <th>483</th>
      <th>484</th>
      <th>485</th>
      <th>486</th>
      <th>487</th>
      <th>488</th>
      <th>489</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
      <th>500</th>
      <th>501</th>
      <th>502</th>
      <th>503</th>
      <th>504</th>
      <th>505</th>
      <th>506</th>
      <th>507</th>
      <th>508</th>
      <th>509</th>
      <th>510</th>
      <th>511</th>
      <th>512</th>
      <th>513</th>
      <th>514</th>
      <th>515</th>
      <th>516</th>
      <th>517</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.208784</td>
      <td>0.452353</td>
      <td>-0.008553</td>
      <td>0.056482</td>
      <td>0.079044</td>
      <td>0.017718</td>
      <td>0.071164</td>
      <td>0.152168</td>
      <td>0.090291</td>
      <td>0.129122</td>
      <td>0.047420</td>
      <td>0.033249</td>
      <td>0.810197</td>
      <td>-0.289042</td>
      <td>-0.145634</td>
      <td>0.423180</td>
      <td>-0.455556</td>
      <td>-1.671116</td>
      <td>-1.144862</td>
      <td>-1.286545</td>
      <td>-0.186224</td>
      <td>-0.299154</td>
      <td>0.336056</td>
      <td>-0.119001</td>
      <td>2.665005</td>
      <td>1.614101</td>
      <td>-0.164951</td>
      <td>-0.320927</td>
      <td>-0.393423</td>
      <td>-0.318194</td>
      <td>-0.213805</td>
      <td>-0.047518</td>
      <td>-0.765763</td>
      <td>-0.857419</td>
      <td>-0.459850</td>
      <td>1.147505</td>
      <td>2.457595</td>
      <td>1.652448</td>
      <td>0.100004</td>
      <td>-0.076198</td>
      <td>...</td>
      <td>0.002324</td>
      <td>-0.707736</td>
      <td>-0.263574</td>
      <td>-0.224117</td>
      <td>0.987561</td>
      <td>0.182526</td>
      <td>0.941346</td>
      <td>0.696081</td>
      <td>1.279680</td>
      <td>-0.335420</td>
      <td>0.923162</td>
      <td>0.126354</td>
      <td>0.812596</td>
      <td>0.717914</td>
      <td>1.333886</td>
      <td>1.302365</td>
      <td>0.511569</td>
      <td>1.208792</td>
      <td>1.521369</td>
      <td>0.967858</td>
      <td>-0.504956</td>
      <td>1.426932</td>
      <td>0.274355</td>
      <td>0.323328</td>
      <td>1.134257</td>
      <td>-0.467784</td>
      <td>-3.022798</td>
      <td>-1.318924</td>
      <td>-0.226649</td>
      <td>-1.594439</td>
      <td>-1.200560</td>
      <td>-1.537657</td>
      <td>-1.595043</td>
      <td>-0.266432</td>
      <td>0.425841</td>
      <td>1.007362</td>
      <td>0.874131</td>
      <td>-0.454603</td>
      <td>-0.369213</td>
      <td>0.775698</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.044880</td>
      <td>0.050426</td>
      <td>0.001063</td>
      <td>0.106160</td>
      <td>0.090461</td>
      <td>-0.020371</td>
      <td>0.064405</td>
      <td>0.046439</td>
      <td>-0.012742</td>
      <td>0.040801</td>
      <td>0.031417</td>
      <td>0.035625</td>
      <td>0.637811</td>
      <td>-0.110113</td>
      <td>-0.322858</td>
      <td>-0.088839</td>
      <td>-1.716767</td>
      <td>-1.441497</td>
      <td>-0.531687</td>
      <td>0.583716</td>
      <td>-0.905734</td>
      <td>-0.813934</td>
      <td>0.154703</td>
      <td>-0.544165</td>
      <td>-0.298108</td>
      <td>-0.297577</td>
      <td>-0.296125</td>
      <td>-0.480501</td>
      <td>-0.423014</td>
      <td>-0.025600</td>
      <td>1.763173</td>
      <td>2.156390</td>
      <td>0.850571</td>
      <td>-0.004497</td>
      <td>0.412165</td>
      <td>-0.092449</td>
      <td>-0.188827</td>
      <td>-0.197095</td>
      <td>-0.321209</td>
      <td>-0.507211</td>
      <td>...</td>
      <td>-1.030474</td>
      <td>-0.293628</td>
      <td>-0.216487</td>
      <td>-0.015757</td>
      <td>0.177297</td>
      <td>0.767854</td>
      <td>-0.638702</td>
      <td>1.493421</td>
      <td>-0.300535</td>
      <td>-0.052967</td>
      <td>0.208804</td>
      <td>0.726020</td>
      <td>-0.484535</td>
      <td>1.364113</td>
      <td>-0.432938</td>
      <td>0.500661</td>
      <td>0.770768</td>
      <td>1.418181</td>
      <td>-0.047990</td>
      <td>1.954450</td>
      <td>0.810134</td>
      <td>0.577882</td>
      <td>-0.302202</td>
      <td>0.269032</td>
      <td>-1.814091</td>
      <td>1.249080</td>
      <td>1.444967</td>
      <td>-0.516551</td>
      <td>-1.452787</td>
      <td>-1.271129</td>
      <td>-0.938948</td>
      <td>-1.210805</td>
      <td>-0.675913</td>
      <td>-0.299409</td>
      <td>0.463112</td>
      <td>0.975158</td>
      <td>0.643107</td>
      <td>-0.454603</td>
      <td>-0.491744</td>
      <td>1.064853</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.002712</td>
      <td>-0.024945</td>
      <td>-0.061548</td>
      <td>0.021789</td>
      <td>0.109332</td>
      <td>0.046419</td>
      <td>-0.070494</td>
      <td>-0.081502</td>
      <td>0.063099</td>
      <td>0.092097</td>
      <td>-0.008355</td>
      <td>0.028350</td>
      <td>-0.135179</td>
      <td>0.696900</td>
      <td>-1.297672</td>
      <td>-1.605830</td>
      <td>-1.556359</td>
      <td>-1.560414</td>
      <td>-1.042074</td>
      <td>-0.589041</td>
      <td>0.918043</td>
      <td>0.035097</td>
      <td>-0.406956</td>
      <td>0.058168</td>
      <td>0.016407</td>
      <td>0.709248</td>
      <td>0.018641</td>
      <td>-0.454507</td>
      <td>-0.100879</td>
      <td>-0.117865</td>
      <td>0.443244</td>
      <td>0.531742</td>
      <td>1.327525</td>
      <td>0.595293</td>
      <td>0.166335</td>
      <td>-0.154833</td>
      <td>0.072337</td>
      <td>0.562125</td>
      <td>0.059394</td>
      <td>-0.269650</td>
      <td>...</td>
      <td>-1.375322</td>
      <td>-0.218907</td>
      <td>0.034549</td>
      <td>-0.597282</td>
      <td>-1.486165</td>
      <td>-0.471921</td>
      <td>0.196878</td>
      <td>0.261671</td>
      <td>-0.578786</td>
      <td>-0.611351</td>
      <td>-1.546994</td>
      <td>-0.292636</td>
      <td>0.181758</td>
      <td>0.279534</td>
      <td>-0.545264</td>
      <td>-0.167451</td>
      <td>-0.207293</td>
      <td>-0.517863</td>
      <td>1.269226</td>
      <td>1.385607</td>
      <td>-2.728832</td>
      <td>2.228279</td>
      <td>0.209649</td>
      <td>-0.838638</td>
      <td>0.002533</td>
      <td>-0.517197</td>
      <td>-1.898425</td>
      <td>-0.610996</td>
      <td>-1.182017</td>
      <td>-0.372433</td>
      <td>-1.561604</td>
      <td>-1.461126</td>
      <td>-1.162622</td>
      <td>-0.254636</td>
      <td>-0.034719</td>
      <td>0.011006</td>
      <td>-0.065367</td>
      <td>-0.454603</td>
      <td>-0.335249</td>
      <td>0.167182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.101044</td>
      <td>-0.044147</td>
      <td>0.186567</td>
      <td>-0.026460</td>
      <td>0.071382</td>
      <td>0.056075</td>
      <td>0.088195</td>
      <td>0.081984</td>
      <td>0.043355</td>
      <td>0.207549</td>
      <td>0.201431</td>
      <td>0.035981</td>
      <td>-1.865851</td>
      <td>-0.636383</td>
      <td>-1.835063</td>
      <td>0.671720</td>
      <td>-0.181450</td>
      <td>-0.908812</td>
      <td>0.682990</td>
      <td>-1.068005</td>
      <td>-0.904029</td>
      <td>-1.376660</td>
      <td>0.538679</td>
      <td>0.435616</td>
      <td>-0.333716</td>
      <td>0.485762</td>
      <td>-0.114997</td>
      <td>-0.338853</td>
      <td>-0.781394</td>
      <td>0.600875</td>
      <td>2.386129</td>
      <td>1.238794</td>
      <td>0.240983</td>
      <td>0.000546</td>
      <td>0.475923</td>
      <td>0.053255</td>
      <td>-0.189086</td>
      <td>0.407508</td>
      <td>-0.059326</td>
      <td>-0.219881</td>
      <td>...</td>
      <td>-1.250265</td>
      <td>1.540934</td>
      <td>-0.630279</td>
      <td>-1.668173</td>
      <td>-1.475965</td>
      <td>-0.038551</td>
      <td>-1.314430</td>
      <td>0.666956</td>
      <td>0.486482</td>
      <td>-1.664254</td>
      <td>-1.558035</td>
      <td>-0.096544</td>
      <td>-1.157187</td>
      <td>0.485260</td>
      <td>0.421423</td>
      <td>0.271677</td>
      <td>-1.700334</td>
      <td>0.335784</td>
      <td>-0.417967</td>
      <td>0.810502</td>
      <td>1.820607</td>
      <td>-0.120928</td>
      <td>-0.513332</td>
      <td>-0.023208</td>
      <td>-2.007936</td>
      <td>2.594519</td>
      <td>0.746281</td>
      <td>-1.121204</td>
      <td>0.046623</td>
      <td>-0.287477</td>
      <td>-0.655213</td>
      <td>-0.595296</td>
      <td>-1.299392</td>
      <td>-0.090255</td>
      <td>0.385908</td>
      <td>0.758699</td>
      <td>0.889532</td>
      <td>-0.454603</td>
      <td>0.107731</td>
      <td>0.018234</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.019632</td>
      <td>-0.035814</td>
      <td>-0.013339</td>
      <td>-0.000614</td>
      <td>0.031972</td>
      <td>-0.018958</td>
      <td>-0.033153</td>
      <td>-0.074345</td>
      <td>-0.021467</td>
      <td>-0.133378</td>
      <td>-0.073489</td>
      <td>-0.039377</td>
      <td>0.350352</td>
      <td>0.994860</td>
      <td>0.677509</td>
      <td>0.111521</td>
      <td>0.506647</td>
      <td>1.117271</td>
      <td>0.833085</td>
      <td>0.827771</td>
      <td>0.505643</td>
      <td>0.641495</td>
      <td>1.090435</td>
      <td>0.916548</td>
      <td>-0.665909</td>
      <td>-0.084322</td>
      <td>0.172613</td>
      <td>-0.884541</td>
      <td>-0.909661</td>
      <td>-0.088299</td>
      <td>0.483259</td>
      <td>0.517938</td>
      <td>-0.055752</td>
      <td>0.629438</td>
      <td>0.965764</td>
      <td>0.032395</td>
      <td>-0.500937</td>
      <td>-0.235502</td>
      <td>-0.057003</td>
      <td>-0.680940</td>
      <td>...</td>
      <td>-0.316716</td>
      <td>0.774707</td>
      <td>-0.146151</td>
      <td>0.906008</td>
      <td>0.526437</td>
      <td>-0.335985</td>
      <td>-1.340416</td>
      <td>0.397575</td>
      <td>-0.874956</td>
      <td>0.803713</td>
      <td>0.562105</td>
      <td>-0.363817</td>
      <td>-1.235536</td>
      <td>0.297470</td>
      <td>-0.808743</td>
      <td>-0.573286</td>
      <td>-2.025075</td>
      <td>-0.348783</td>
      <td>-0.848667</td>
      <td>-0.164191</td>
      <td>-0.604540</td>
      <td>0.450336</td>
      <td>-0.424901</td>
      <td>0.100236</td>
      <td>0.105479</td>
      <td>0.468032</td>
      <td>-0.279087</td>
      <td>0.349954</td>
      <td>0.406174</td>
      <td>0.172529</td>
      <td>-0.098756</td>
      <td>0.151932</td>
      <td>-0.085646</td>
      <td>-0.143811</td>
      <td>0.481748</td>
      <td>-0.169453</td>
      <td>-0.111572</td>
      <td>-0.265064</td>
      <td>-0.008018</td>
      <td>-0.341557</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 518 columns</p>
</div>




```python
## Implement PCA
pca_7 = PCA()
pca_7.fit(df_scale2)

## Cumulated Variance and extract the number of components which cumulate > 70% of total variance.
cumvar_7 = np.cumsum(pca_7.explained_variance_ratio_)
cumvar_7 = cumvar_7[cumvar_7 <= 0.70].tolist()
comp_7 = len(cumvar_7) + 1
```


```python
## Explained Variance Ratio
np.cumsum(pca_7.explained_variance_ratio_)[0:100]
```




    array([0.11981209, 0.18783652, 0.22978658, 0.26341166, 0.29309699,
           0.32096036, 0.34615863, 0.37018642, 0.39288233, 0.4139717 ,
           0.43203019, 0.44786818, 0.46229333, 0.47595843, 0.48892151,
           0.50137491, 0.51216316, 0.52287975, 0.53301808, 0.54217426,
           0.55077623, 0.55930237, 0.56732985, 0.57528061, 0.58255467,
           0.58971868, 0.59659928, 0.60317865, 0.60969622, 0.61583576,
           0.62180412, 0.62758712, 0.63305729, 0.63837313, 0.64342951,
           0.64839667, 0.65320178, 0.65792701, 0.66255006, 0.66708503,
           0.67142046, 0.67570121, 0.67990594, 0.68405179, 0.6881781 ,
           0.69212525, 0.69601766, 0.69979194, 0.70346478, 0.70709962,
           0.71063087, 0.71410517, 0.71743727, 0.72073475, 0.72397094,
           0.72711942, 0.73019727, 0.73325445, 0.73619701, 0.73909631,
           0.74198861, 0.74482125, 0.74759659, 0.75030818, 0.75298924,
           0.75565874, 0.75828488, 0.76086094, 0.76343457, 0.76593635,
           0.76839782, 0.77084772, 0.77326604, 0.77565001, 0.77800713,
           0.78035852, 0.78269297, 0.78499351, 0.78723414, 0.78945422,
           0.79166112, 0.79386078, 0.79603726, 0.79819067, 0.80032888,
           0.8024381 , 0.80452173, 0.80658498, 0.80862996, 0.81065999,
           0.81266855, 0.81464916, 0.81660003, 0.81852961, 0.82044162,
           0.82232257, 0.82419309, 0.82604758, 0.82787576, 0.82968369])




```python
## Result
print('70% of variance is caught by', comp_7, 'components')
```

    70% of variance is caught by 49 components
    


```python
## Plot of Cumulative Variance
figure(figsize = (12, 6), dpi = 80)
plt.plot(np.cumsum(pca_7.explained_variance_ratio_))
plt.axvline(comp_7, color = 'red');
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.xlim((0,100))
print('As we can see, the 70% of variance is caught by', comp_7, 'components.')
```

    As we can see, the 70% of variance is caught by 49 components.
    


    
![png](output_209_1.png)
    


Extract Score Z.


```python
## Extract Score
Z_7 = pca_7.transform(df_scale2)
Z_7 = pd.DataFrame(Z_7)
## We extract the projection of X in the principal components.

## Extract first 64 cocmponents.
Z_7 = Z_7.iloc[0:, :comp_7]
```


```python
## Check
Z_7.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
      <th>32</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
      <th>44</th>
      <th>45</th>
      <th>46</th>
      <th>47</th>
      <th>48</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.060070</td>
      <td>-3.089439</td>
      <td>-5.565923</td>
      <td>0.080540</td>
      <td>2.309270</td>
      <td>-3.597832</td>
      <td>-5.054209</td>
      <td>-0.889464</td>
      <td>-5.108313</td>
      <td>-4.067441</td>
      <td>3.572938</td>
      <td>-1.159990</td>
      <td>-2.207847</td>
      <td>-1.590584</td>
      <td>0.223469</td>
      <td>0.433007</td>
      <td>-1.436558</td>
      <td>1.154625</td>
      <td>-0.077387</td>
      <td>1.605499</td>
      <td>-1.225731</td>
      <td>-0.316250</td>
      <td>0.250172</td>
      <td>-2.847029</td>
      <td>0.360655</td>
      <td>-3.041298</td>
      <td>0.892076</td>
      <td>1.449698</td>
      <td>-0.321163</td>
      <td>0.564567</td>
      <td>1.440019</td>
      <td>-0.743921</td>
      <td>-0.548346</td>
      <td>-0.478788</td>
      <td>-0.800454</td>
      <td>0.595390</td>
      <td>1.367694</td>
      <td>-0.865034</td>
      <td>-0.115469</td>
      <td>0.701893</td>
      <td>-2.255730</td>
      <td>0.352275</td>
      <td>-1.812992</td>
      <td>-0.385806</td>
      <td>-1.353177</td>
      <td>-0.144009</td>
      <td>-0.795262</td>
      <td>1.588618</td>
      <td>0.892265</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.509246</td>
      <td>-5.384927</td>
      <td>-4.332572</td>
      <td>-1.113460</td>
      <td>-3.068300</td>
      <td>2.204037</td>
      <td>-0.234569</td>
      <td>4.726578</td>
      <td>-0.604041</td>
      <td>-3.417902</td>
      <td>2.969570</td>
      <td>0.127854</td>
      <td>-0.722160</td>
      <td>0.904164</td>
      <td>-1.056876</td>
      <td>-2.531507</td>
      <td>-0.230250</td>
      <td>0.510513</td>
      <td>-2.259663</td>
      <td>0.534653</td>
      <td>-0.161708</td>
      <td>-1.717528</td>
      <td>1.427582</td>
      <td>-3.051052</td>
      <td>0.573445</td>
      <td>-2.698419</td>
      <td>-0.714682</td>
      <td>1.074101</td>
      <td>-0.859668</td>
      <td>2.312505</td>
      <td>-0.855573</td>
      <td>0.707546</td>
      <td>0.354507</td>
      <td>-1.060272</td>
      <td>1.844368</td>
      <td>-1.362443</td>
      <td>2.580939</td>
      <td>-1.632471</td>
      <td>1.159818</td>
      <td>0.914117</td>
      <td>-1.623623</td>
      <td>1.195638</td>
      <td>-0.955477</td>
      <td>0.405992</td>
      <td>-0.122113</td>
      <td>0.697013</td>
      <td>-0.585097</td>
      <td>2.636847</td>
      <td>-0.177606</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.954593</td>
      <td>-3.178022</td>
      <td>-3.494718</td>
      <td>-0.934800</td>
      <td>-1.098688</td>
      <td>0.455750</td>
      <td>-2.390566</td>
      <td>1.266916</td>
      <td>0.985001</td>
      <td>-1.733520</td>
      <td>2.806514</td>
      <td>-2.365280</td>
      <td>-0.655932</td>
      <td>0.470489</td>
      <td>0.112723</td>
      <td>-0.440214</td>
      <td>-1.329598</td>
      <td>0.536812</td>
      <td>0.613815</td>
      <td>2.531022</td>
      <td>0.213038</td>
      <td>-0.983676</td>
      <td>-0.159938</td>
      <td>-2.368912</td>
      <td>-3.419339</td>
      <td>-1.974347</td>
      <td>2.764438</td>
      <td>1.807662</td>
      <td>-1.772858</td>
      <td>-0.106143</td>
      <td>2.107336</td>
      <td>0.077253</td>
      <td>-0.829437</td>
      <td>-0.471847</td>
      <td>2.604790</td>
      <td>0.328109</td>
      <td>0.566734</td>
      <td>-1.895733</td>
      <td>1.725603</td>
      <td>1.045803</td>
      <td>-0.848437</td>
      <td>0.523569</td>
      <td>-0.615561</td>
      <td>0.542990</td>
      <td>-1.050345</td>
      <td>0.036632</td>
      <td>-1.478038</td>
      <td>1.834811</td>
      <td>-0.275641</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.695430</td>
      <td>1.329784</td>
      <td>-5.129153</td>
      <td>-0.915313</td>
      <td>-2.155916</td>
      <td>-0.120454</td>
      <td>0.072621</td>
      <td>3.904953</td>
      <td>0.213868</td>
      <td>-2.841647</td>
      <td>1.364953</td>
      <td>1.260873</td>
      <td>0.323507</td>
      <td>2.791101</td>
      <td>2.677146</td>
      <td>-1.369212</td>
      <td>-0.696572</td>
      <td>0.325820</td>
      <td>-0.943323</td>
      <td>1.015017</td>
      <td>-2.506290</td>
      <td>-2.761328</td>
      <td>0.190591</td>
      <td>0.912299</td>
      <td>-1.293208</td>
      <td>0.417755</td>
      <td>-0.336510</td>
      <td>-2.035364</td>
      <td>3.589013</td>
      <td>-1.083368</td>
      <td>2.753534</td>
      <td>1.246875</td>
      <td>0.050735</td>
      <td>1.078600</td>
      <td>2.261801</td>
      <td>0.200077</td>
      <td>1.084182</td>
      <td>0.234084</td>
      <td>2.954903</td>
      <td>0.206754</td>
      <td>-0.037722</td>
      <td>-0.153328</td>
      <td>0.113327</td>
      <td>0.843546</td>
      <td>-0.054427</td>
      <td>-0.498017</td>
      <td>-0.674816</td>
      <td>-1.253862</td>
      <td>-0.908248</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.870800</td>
      <td>-1.123613</td>
      <td>2.985175</td>
      <td>-0.830491</td>
      <td>-2.443379</td>
      <td>4.542687</td>
      <td>-0.159431</td>
      <td>-1.837118</td>
      <td>1.026206</td>
      <td>-2.033920</td>
      <td>-0.940295</td>
      <td>-2.119117</td>
      <td>1.312825</td>
      <td>5.708031</td>
      <td>-0.101215</td>
      <td>0.445861</td>
      <td>1.724467</td>
      <td>-0.290227</td>
      <td>0.503632</td>
      <td>-1.364637</td>
      <td>1.843659</td>
      <td>1.081822</td>
      <td>1.090737</td>
      <td>0.447308</td>
      <td>1.157101</td>
      <td>1.097130</td>
      <td>-1.372946</td>
      <td>1.356532</td>
      <td>1.129801</td>
      <td>0.238177</td>
      <td>-1.081960</td>
      <td>-1.342903</td>
      <td>0.115791</td>
      <td>-1.200921</td>
      <td>0.533756</td>
      <td>-1.740978</td>
      <td>0.746127</td>
      <td>-2.422150</td>
      <td>-1.195532</td>
      <td>1.322819</td>
      <td>0.255863</td>
      <td>2.031239</td>
      <td>-0.617361</td>
      <td>-1.745593</td>
      <td>-0.134674</td>
      <td>-0.460643</td>
      <td>-0.503801</td>
      <td>-0.530340</td>
      <td>-0.390840</td>
    </tr>
  </tbody>
</table>
</div>



Find the optimal number of clusters.

Using only the previously defined data and criteria, and the *kmeans++* method, the optimal number the optimal *k* is *12*, with a score of *26747428.658*.


```python
## Metric = Distortion, which computes the sum of squared distances from each point to its assigned center. 
figure(figsize = (12, 6), dpi = 80)
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'distortion', timings = False)
visualizer.fit(Z_7)
visualizer.show()
```


    
![png](output_215_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94d2060850>




```python
## Metric = Calinski-Harabaszion, which score computes the ratio of dispersion between and within clusters.
figure(figsize = (12, 6), dpi = 80)
visualizer = KElbowVisualizer(model, k = (2,20), metric = 'calinski_harabasz', timings = False)
visualizer.fit(Z_7)
visualizer.show()
```


    
![png](output_216_0.png)
    





    <matplotlib.axes._subplots.AxesSubplot at 0x7f94d228dad0>




```python
## Run K-Means++ with k = 8
clusters_8 = KMeans(init = 'k-means++', n_clusters = 8)
clusters_8.fit(Z_7)
```




    KMeans()



Characterize your clusters using 5-10 variables.

Now we have characterized our cluster with five variables: track_duration as before, track_bitrate: it is a variable that is important for the quality of the audio. Then we have chosen chroma_cens, that is the class of tonality rmse, that is root mean squared error of the song’s energy1 and tonnetz. that is representation of the scale range of just intonation.


```python
## Extract from second dataset
d_pivot = df2.loc[:, ('chroma_cens_median_01','rmse_median_01','tonnetz_median_01')]
```


```python
# Extract track duration
d_pivot['track_duration'] = df3.loc[:, ('track_duration')]
```


```python
## Extract bitrate
d_pivot['track_bit_rate'] = df3.loc[:, ('track_bit_rate')]
```


```python
## Check
d_pivot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chroma_cens_median_01</th>
      <th>rmse_median_01</th>
      <th>tonnetz_median_01</th>
      <th>track_duration</th>
      <th>track_bit_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.482825</td>
      <td>2.653895</td>
      <td>-0.003789</td>
      <td>168</td>
      <td>256000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.230579</td>
      <td>3.706424</td>
      <td>-0.000108</td>
      <td>237</td>
      <td>256000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.255472</td>
      <td>2.409692</td>
      <td>-0.007385</td>
      <td>206</td>
      <td>256000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.230555</td>
      <td>3.756495</td>
      <td>-0.021108</td>
      <td>161</td>
      <td>192000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.200830</td>
      <td>4.367132</td>
      <td>0.011057</td>
      <td>311</td>
      <td>256000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Discrete each variable in 4 groups
for i in d_pivot.columns:
    d_pivot[i] = pd.qcut(d_pivot[i], q = 4)
```

Pivot table for Duration.


```python
## Duration
pv_d = pd.crosstab(d_pivot.track_duration, clusters_8.labels_,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_d = pv_d.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8'}, axis=1)
## Index Row
pv_d.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_d = pv_d.append(pv_d.sum().rename('Total'))
```


```python
## Check
pv_d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>23.42</td>
      <td>16.79</td>
      <td>29.41</td>
      <td>15.34</td>
      <td>48.81</td>
      <td>29.39</td>
      <td>40.0</td>
      <td>56.02</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>20.85</td>
      <td>24.32</td>
      <td>28.62</td>
      <td>23.76</td>
      <td>16.67</td>
      <td>26.54</td>
      <td>20.0</td>
      <td>16.34</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>23.04</td>
      <td>29.99</td>
      <td>23.00</td>
      <td>29.84</td>
      <td>11.90</td>
      <td>22.22</td>
      <td>20.0</td>
      <td>13.74</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>32.68</td>
      <td>28.90</td>
      <td>18.97</td>
      <td>31.06</td>
      <td>22.62</td>
      <td>21.85</td>
      <td>20.0</td>
      <td>13.90</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.0</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



Pivot table for Track Bitrate.


```python
## Track Bitrate
pv_t = pd.crosstab(d_pivot.track_bit_rate, clusters_8.labels_,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_t = pv_t.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8'}, axis=1)
## Index Row
pv_t.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_t = pv_t.append(pv_t.sum().rename('Total'))
```


```python
## Check
pv_t
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>25.84</td>
      <td>20.88</td>
      <td>25.62</td>
      <td>24.40</td>
      <td>15.48</td>
      <td>29.48</td>
      <td>0.0</td>
      <td>25.63</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>22.76</td>
      <td>19.69</td>
      <td>29.40</td>
      <td>21.36</td>
      <td>30.95</td>
      <td>26.75</td>
      <td>60.0</td>
      <td>24.76</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>50.96</td>
      <td>58.96</td>
      <td>44.49</td>
      <td>53.71</td>
      <td>52.38</td>
      <td>43.42</td>
      <td>40.0</td>
      <td>48.78</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>0.44</td>
      <td>0.47</td>
      <td>0.48</td>
      <td>0.53</td>
      <td>1.19</td>
      <td>0.35</td>
      <td>0.0</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.0</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



The category that is **less discriminating** for clustering observations is the **fourth**, which contributes nothing to all clusters.

Pivot table for tonnetz


```python
## Tonnetz
pv_to = pd.crosstab(d_pivot.tonnetz_median_01, clusters_8.labels_,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_to = pv_to.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8'}, axis=1)
## Index Row
pv_to.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_to = pv_to.append(pv_to.sum().rename('Total'))
```


```python
## Check
pv_to
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>25.95</td>
      <td>32.39</td>
      <td>20.63</td>
      <td>23.85</td>
      <td>52.38</td>
      <td>23.91</td>
      <td>0.0</td>
      <td>30.78</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>25.29</td>
      <td>23.27</td>
      <td>25.31</td>
      <td>32.95</td>
      <td>10.71</td>
      <td>17.63</td>
      <td>100.0</td>
      <td>24.10</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>25.17</td>
      <td>23.16</td>
      <td>28.29</td>
      <td>27.25</td>
      <td>9.52</td>
      <td>20.53</td>
      <td>0.0</td>
      <td>22.36</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>23.59</td>
      <td>21.17</td>
      <td>25.77</td>
      <td>15.94</td>
      <td>27.38</td>
      <td>37.94</td>
      <td>0.0</td>
      <td>22.77</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>99.99</td>
      <td>100.01</td>
      <td>100.0</td>
      <td>100.01</td>
    </tr>
  </tbody>
</table>
</div>



The cluster that best discretes the categories is the **seventh**, which groups **100%** of the observations from the **second** category.

Pivot table for rmse


```python
## Rmse
pv_r = pd.crosstab(d_pivot.rmse_median_01, clusters_8.labels_,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_r = pv_r.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8'}, axis=1)
## Index Row
pv_r.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_r = pv_r.append(pv_r.sum().rename('Total'))
```


```python
## Check
pv_r
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>52.14</td>
      <td>14.37</td>
      <td>14.43</td>
      <td>22.23</td>
      <td>21.43</td>
      <td>35.67</td>
      <td>80.0</td>
      <td>24.24</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>26.02</td>
      <td>22.41</td>
      <td>23.65</td>
      <td>23.31</td>
      <td>19.05</td>
      <td>31.18</td>
      <td>0.0</td>
      <td>20.33</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>15.46</td>
      <td>29.06</td>
      <td>27.46</td>
      <td>28.03</td>
      <td>22.62</td>
      <td>21.55</td>
      <td>0.0</td>
      <td>20.66</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>6.38</td>
      <td>34.17</td>
      <td>34.46</td>
      <td>26.43</td>
      <td>36.90</td>
      <td>11.59</td>
      <td>20.0</td>
      <td>34.77</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.0</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



Pivot table for chroma


```python
## Chroma
pv_c = pd.crosstab(d_pivot.chroma_cens_median_01, clusters_8.labels_,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_c = pv_c.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8'}, axis=1)
## Index Row
pv_c.index = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
## Add Total
pv_c = pv_c.append(pv_c.sum().rename('Total'))
```


```python
## Check
pv_c
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 7</th>
      <th>Cluster 8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Category 1</th>
      <td>39.20</td>
      <td>34.40</td>
      <td>15.34</td>
      <td>16.84</td>
      <td>54.76</td>
      <td>34.18</td>
      <td>80.0</td>
      <td>11.91</td>
    </tr>
    <tr>
      <th>Category 2</th>
      <td>23.73</td>
      <td>22.78</td>
      <td>29.05</td>
      <td>25.17</td>
      <td>8.33</td>
      <td>23.77</td>
      <td>20.0</td>
      <td>18.61</td>
    </tr>
    <tr>
      <th>Category 3</th>
      <td>17.90</td>
      <td>16.11</td>
      <td>31.42</td>
      <td>31.74</td>
      <td>4.76</td>
      <td>19.04</td>
      <td>0.0</td>
      <td>30.91</td>
    </tr>
    <tr>
      <th>Category 4</th>
      <td>19.17</td>
      <td>26.71</td>
      <td>24.19</td>
      <td>26.25</td>
      <td>32.14</td>
      <td>23.01</td>
      <td>0.0</td>
      <td>38.57</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.0</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>



Pivot table for genres


```python
d_pivot['song_genre'] = song_genre
```


```python
## Genre
pv_g = pd.crosstab(d_pivot.song_genre, clusters_8.labels_,  normalize = 'columns')\
.round(4)*100
```


```python
## Index Col
pv_g = pv_g.rename({0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3', 3: 'Cluster 4', 4: 'Cluster 5', 5: 'Cluster 6', 6: 'Cluster 7',
                    7: 'Cluster 8'}, axis=1)
## Add Total
pv_g = pv_g.append(pv_g.sum().rename('Total'))
```


```python
## Check
pv_g
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Cluster 1</th>
      <th>Cluster 2</th>
      <th>Cluster 3</th>
      <th>Cluster 4</th>
      <th>Cluster 5</th>
      <th>Cluster 6</th>
      <th>Cluster 8</th>
    </tr>
    <tr>
      <th>song_genre</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Blues</th>
      <td>0.39</td>
      <td>0.23</td>
      <td>0.11</td>
      <td>0.18</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Classical</th>
      <td>14.08</td>
      <td>0.39</td>
      <td>0.11</td>
      <td>0.23</td>
      <td>0.00</td>
      <td>6.01</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Country</th>
      <td>0.30</td>
      <td>0.33</td>
      <td>0.45</td>
      <td>0.30</td>
      <td>0.00</td>
      <td>0.58</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>Easy Listening</th>
      <td>0.07</td>
      <td>0.04</td>
      <td>0.01</td>
      <td>0.12</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Electronic</th>
      <td>13.25</td>
      <td>28.56</td>
      <td>11.99</td>
      <td>33.78</td>
      <td>34.38</td>
      <td>7.85</td>
      <td>18.04</td>
    </tr>
    <tr>
      <th>Experimental</th>
      <td>28.08</td>
      <td>18.45</td>
      <td>16.44</td>
      <td>21.61</td>
      <td>25.00</td>
      <td>22.95</td>
      <td>39.02</td>
    </tr>
    <tr>
      <th>Folk</th>
      <td>11.73</td>
      <td>4.05</td>
      <td>1.67</td>
      <td>1.82</td>
      <td>0.00</td>
      <td>16.61</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>Hip-Hop</th>
      <td>1.34</td>
      <td>9.56</td>
      <td>4.86</td>
      <td>17.61</td>
      <td>0.00</td>
      <td>0.77</td>
      <td>2.33</td>
    </tr>
    <tr>
      <th>Instrumental</th>
      <td>9.57</td>
      <td>4.68</td>
      <td>1.86</td>
      <td>1.61</td>
      <td>18.75</td>
      <td>8.49</td>
      <td>3.35</td>
    </tr>
    <tr>
      <th>International</th>
      <td>5.39</td>
      <td>1.58</td>
      <td>1.28</td>
      <td>3.69</td>
      <td>0.00</td>
      <td>4.68</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>Jazz</th>
      <td>3.69</td>
      <td>1.16</td>
      <td>0.45</td>
      <td>1.15</td>
      <td>0.00</td>
      <td>1.41</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>Old-Time / Historic</th>
      <td>1.01</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.48</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Pop</th>
      <td>4.79</td>
      <td>5.15</td>
      <td>3.76</td>
      <td>4.66</td>
      <td>18.75</td>
      <td>5.41</td>
      <td>6.21</td>
    </tr>
    <tr>
      <th>Rock</th>
      <td>5.44</td>
      <td>25.15</td>
      <td>56.27</td>
      <td>10.17</td>
      <td>0.00</td>
      <td>18.59</td>
      <td>28.71</td>
    </tr>
    <tr>
      <th>Soul-RnB</th>
      <td>0.23</td>
      <td>0.40</td>
      <td>0.32</td>
      <td>0.75</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Spoken</th>
      <td>0.62</td>
      <td>0.23</td>
      <td>0.39</td>
      <td>2.33</td>
      <td>3.12</td>
      <td>0.65</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>99.98</td>
      <td>99.99</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>100.00</td>
      <td>100.01</td>
      <td>99.99</td>
    </tr>
  </tbody>
</table>
</div>



If you could choose, would you rather collect more observations (with fewer features) or fewer observations (with more features) based on the previous analyses?

We will collect more observations and less variables to overcome the problem of the *curse of dimensionality*. In fact, the curse of dimensionality (*peak phenomenon*),  also known as *Hughes phenomenon*. This phenomenon states that with a fixed number of training samples, the average predictive power (expected) of a classifier or regressor first increases as the number of dimensions or characteristics used increases but beyond a certain dimensionality begins to deteriorate instead of improving steadily.

![0_iogC18To3T5GVQBb.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfIAAAFRCAIAAAAAe7jHAABiH0lEQVR42uy9d1wU1/f/P7v03puAdFCaioIFiEYUO2pExYpG7L0TS+yxawSNimgi2CNqVMRCUQGlC4IgSC/Se9/C/h4/7tv7nc+q6wK7sMB5/sFj7s7ulDOX19w5c+45wiwWiwAAAAB6ClQwAQAAAMg6AAAAALIOAAAAgKwDAAAAIOsAAAAg6wAAAADIOgAAAACyDgAAAICsAwAAACDrAAAAIOsAAAAAyDp/iI6O/uuvv2pra+EiAQAAcI9wlx8Bk8msqanBzVevXqGFuLi4+Ph4Z2dnGRkZuE4AAABcQunyDI6pqalbt2793lptbW1h4f/dezZt2qSjo4NX7d69G98PRERETpw4gVeVlJQcOnQINw0NDdeuXYubUVFR169fx83x48dPmDABN+/cuRMeHo6bK1eu7NevH27+8ccfxcXFuHny5El8eI2Nje7u7niVpqbmtm3bcDMxMdHb2xs3bW1tZ82ahZuPHj0KDAzETVdXVysrK9w8ffp0dnY2bu7Zs0dRURE3169fj5cVFRX37NmDmzk5OadOncJNKysrV1dX3AwMDHz06BFuTp8+fdSoUbjp7e2dmJiIm9u2bdPU1MRNd3f3xsZGbozfv3//FStWcGn869evR0VF4ebatWsNDQ25MX5tbe2uXbu4NP7IkSN/+eWX9hn/4MGD5HEGB+Onp6d7enpyafxZs2bZ2tri5oULF1JSUrgxvoSExJEjR7g0fnh4+J07d3BzypQpY8aM4dL4+/btq6io4Mb4urq6Gzdu5NL49+7dwyM5giDc3NwsLCy4MT6Dwdi8eTOXxrexsZk3bx5uBgQEPH36lEvj79y5U1VVlRvjFxQUHDt2DDctLCzc3Ny4NP7Vq1fj4uJw88CBA7Kyst1Y1svKyvz9/dFyc3PzkydP0DKrlQ0bNsjLy6NP+vXrJykpSe4udDr9f74kKnXgwIF4VVNTU3JyMm5KS0sbGxvjZnl5eU5ODhav3377TVtbG6/Ny8srLS0l3xLI9k1OTm5qasLNQYMGUSgU/NiRkJCAV0lKSpLvB9XV1RkZGbipoqJC3mlBQQFZsPT09BQUFMh3vvr6etw0NzcXFRXFTXJvEBUVNTc3x82GhoaPHz/ipoKCgp6eHm4WFxcXFBTgppaWFrn7ZmZmVlVVkQVCQkICNxMSEphMJjfGl5WVJasD2fgEQWi0Qr4PlZeX46aRkRFZQzkYn8FgvH//nkvjq6qqamlptc/4lpaWWM44G7+2tvbTp09cGr9v377KyspkVSI/wnIwvpCQ0IABA7g0fllZWW5uLvnmp6amxqXxk5KSaDQaN8aXkpIyMTHh0vj5+fklJSW4aWBgICcnx43xWSzWu3fvuDS+kpISeURY2AqXxjc1NRUXF+fG+I2NjeT7gby8vL6+PpfGz8rKqqysJN8SREREurGsf4+brXh7e5O1hueIi4tXVVWRLxsAAEC3BiJhAAAAQNZ7ELt27SI/UwMAAHR3hEHWoRMAAACy3qNobGzMaEVeXn7IkCFSUlJgEwAAQNa7GeHh4f7+/u/evXv79m1NTQ1+bywsLDxw4EB7e/vVq1cbGBhA/wAAAGRd0MnPz9+4cePdu3e/uZbBYMS0cu7cuQ0bNuzcubMj0aMAAACdj2AFODIYDDxvqKampra2lrcBjvHx8Q4ODuRZFRgREREcBY9RVVU9cuTI4sWLoaMAANBdELhIGKkvkGfc8IT379+PGTMGa7qGhsaqVaucnJyePn2alZXV1NT0+fPn27dvr1y5Ek96Kikp+fXXX1esWIEnIAAAAAg6LEHlxo0bU6ZMKS4u5snW8vPzVVRU8FmvXbsWudS/9+WFCxfiSXQEQUyYMKG2tpYFAAAg8PSWuPVly5bhlAAHDhzw8PDgkEFMU1Pz6tWrkZGReMJxQECAvb3958+fYRwAAAA4YbqeK1eu4FQzq1atIseqHzx4kMFgfPNX1tbWkZGR1tbWqBkfHz927FhysggAAACQ9S6gqqoKZ3ozNjYm5xrkLOsEQaipqb18+XLatGmomZycPHfu3JaWFug3AACArHcZFy9eRJkIhYSEfHx8yJnwuEFSUtLPz8/JyQk1/f39t2/fDv0GAACQ9a6BRqN5eHig5dmzZw8dOpTtC9zkhKFSqdevX8dpoE+cOPHPP/9A1wEAAGS9C7h58yZ+z7lly5avv8Blqi9paemHDx/iWJrly5e/efMGeg8AAAKIYM0yZTKZuKZJVlZWxzeIPekODg6DBg3qyKZ0dXXv3bvn4OBAa2X69OnR0dF9+/aFPgQAAIzWOcl68Bc6LutPnz5NSkpCy9+rq8f5lSkbdnZ2Fy9eRMslJSVOTk7k0i0AAACCQE+ujjRmzJigoCBUQYpclItMO6ojbd26FT8ETJ8+3c/PjzxxCQAAAEbrfOHdu3dI07/nVW83R48enTRpElq+f//+7t27oRsBAACyzne8vLzQgqam5pw5c773tXZUR6JSqTdv3jQzM0PNQ4cO3bt3D3oSAAAg63yExWI9fPgQLS9atIhDDe/2Fb2TkZF59OgRrla+bNmyoqIi6EwAAICs84vY2Fgc1zh16lR+7EJPT+/69evIq15eXu7m5gadCQAAkHV+8d9//6EFDQ2NIUOGcPhmmyJh2HB0dFy1ahVa9vf3v3TpEvQnAABA1vkC9sBMmTKFc5hKR2SdIIhjx46ZmJig5U2bNmVmZkKXAgCgaxGs6UgtLS3Pnj1Dy2lpae3bSHZ2Ng5n5JMHBiMpKenr6ztixAgGg1FXV+fq6vrq1SsqlQodCwAAkHUCFb07f/48r4bqUlJSDg4OnL/cvlemZKytrXfu3Llv3z6CIMLCwk6cOLFt2zboWAAAdBWCNR2JxWLheaFBQUHBwcHtmI6EZyH98ssvfn5+nXM3GjFiRHR0NEEQoqKiMTExOC8YAABAJyNY7gIKhWLxBTU1tXZsoaqq6vXr12gZZ9Pl+yOPsLCvry9K+Uuj0RYsWECj0aBvAQAAss4DAgIC6HQ6yq4+efLkH36/g69MMSYmJseOHUPLCQkJe/bsgb4FAADIOg/Ab1xHjBihpKTUabJOEMTq1avHjh2Llo8dOxYeHg7dCwAAkPWOgtOg//BlKc+hUCh///23goICCulxdXWF/I4AAICsd4jS0tJPnz6hZVtbW25+0vFIGDKampp//fUXWs7IyMA1VAEAAEDW28Pbt2/RgpCQ0Nf17TpB1gmCcGkFLV+8ePHFixfQyQAA6L2yzmKx8r9QU1PT1p9jD4yFhYWMjExXncVff/2lrq6Olt3c3Gpra6GfAQDQS2WdTqev+oK/v3+7ZX3EiBFc/oSHr0wxCgoKeFJVbm4uuGIAAOhMBGs6UktLC85dntAK99OR6HS6nJxcY2MjQRDXr1+fO3cuN79qR3UkLpk3b96NGzfQ8vPnz3GQDAAAAF8RrOQBVCrV2dkZy3RCQgL3v3337h3S9DaN1vmHh4dHUFBQcXExcsUkJSV1oV8IAIDeQ895ZYo9MBoaGrq6ulz+iuevTDFKSkpkVwxvC+8BAAB0hqxXVVWFh4e/efOmtra2urq6q2S9TUN1/sk6KmCNo2K8vLwgKgYAgO4k6y9fvly2bNnp06efP3+ek5OzatWqiIiILpF1LiPWO4ezZ8/idwMQFQMAQLeR9bKysrNnz65atcrd3Z0gCHNz84ULF16/fr3TTiMnJ6egoKAdo/UTJ04wmUz+HRi4YgAA6JaynpqaqqOjM2rUKFxBYty4cTU1NZWVlZ1zGngikri4uJWVFfc/3LVrF0oNxj9++eWX2bNno2UvL6/AwEDodgAACLqsS0lJVVZWkmMlS0pKqqurRUVFO+c0cMyMpaWliIiIoFkZXDEAAHQavHlb2L9/fyEhodOnT+vq6tLp9Ldv3z548MDCwkJKSuqHv01PT9+1axdaZrFYOEgRcfHiRRxUPn/+fA0NDbzqr7/+wrm0cE1qS0tLgiAqKiouX76Mv6mtrY1fXRIEkZSUFBAQgJbt7OyioqJ++uknvPbFixfx8fG46ezsrKenh5tXrlwpLy/HzU2bNgkJCaHl5uZmDw8PvEpVVdXV1RUtKysr//7772vWrEH+IkdHx/3795Mj2V+/fh0ZGYmbU6ZM6devH25ev3798+fPuLls2TI5OTncPH78OF6WlZVdvnw5bhYWFl67dg03+/XrN2XKFNyMiop69eoVbo4ePXrw4MG4+eDBA5xghyAIV1dX8gQCDw+P5ubm//UhYeGNGzfiVWzG19PTw0GrbMZHHjPyu5CAgABcRwVlYtDW1ubG+A0NDefOnfum8VEfu3//Pm4OHjx49OjR7TP+6tWrJSUluTF+Xl7erVu3uDS+o6PjgAEDcPPu3btZWVncGF9MTGzdunVcGj8hIeH58+e4OXLkSBsbGy6Nf/HiRfLEbw7G79Onz7x587g0fnBwcGxsLG5Onz7d0NCQG+MzmcxTp05xaXxzc/MJEybgJors4NL4S5YsUVRU5Mb4JSUlV69exU0jI6Np06ZxafxHjx59/PgRN1euXCktLd3Fsi4mJrZr1y4PD4+XL18SBJGYmDhw4EDyCXNAXFwc/xcxGIzExES8isVi6evr43BvtklDBgYG2L6lpaVoAZUlEhUVxZWjkYOb/EM5OTm81sTEhK1eh5qaGvm3bHcmXV1dZWVl3CTXv6ZSqeQfysrKkn84Y8aMW7duhYWFEQQRERFB7jcEQaioqJB/yxbh3rdvX/InbI8j5B+SFQdZjLy2T58+5LWKiorktSj3JPnL5KKsYmJi5LVGRkbYeYX/vRFsxmebTUY2Prrhkdeqq6uTfWKoMgk3xhcSEuJgfGlpaQ6H1Cbjs50sB+NLSEhwb3x5eXnyWi0tLfKTLgfjs3UGzsaXl5cnryWr1Q+Nb2Bg0NDQwI3x2ToSZ+OrqqqS17JpGQfjUygU7o2Pk3ngXse98dlcDhyMLyYmxmGnnI3fp08fsrejg+F5PJ5lWllZWVZWpqqqSh5Oto+brXAzy7S6uhpfmJCQkFGjRgnmk1FZWZmpqSm6A+no6CQmJsIEJQAAeA7PAhwbGxtv3bqloKBgZGT09u3bwMBAvkaYkCEP8NtaRJTfkTBsYwSctjcnJ2fr1q3Q/wAAEFxZ37hxY0JCAno8UVZWfvDgAdnd1jmyrqGhwU1FJDKdEAlDxtnZeebMmWjZy8srODgYuiAAAIIo6+np6UJCQocPH0bOpiFDhhw+fDg6OprnyRE5yzp6XyrgnDt3TkVFBb05WLRoUVVVFfRCAAAETtYrKyu/fnXDYDA6J24dy3pbPTAEQWzdupV/yQO+iYqKire3N1rOy8tbtWoV9EIAAARO1s3NzXNzc/39/VFEdkVFxc2bN6WkpNCwVJBH6wcOHOhkWScIwsnJyc3NDS2jN8PQEQEA4BW8UTQJCYmdO3ceOXLEy8tLWVm5tLRUQUHht99+64QTyM3NxWnF2jFa7ypOnz4dEhKSkZFBEMSqVavs7OzIMcIAAABdLOtolsGVK1cyMzOLioqUlJTMzc3bUZuCyWQeOXIELefn57dpqC4sLNy/f/+27vHEiRMbN25kC0buBKSlpX19fe3t7ZlMZlVVlaura1BQEDkWGAAAoH3wMjGvlJSUnp6ejY2NgYFBY2NjOxzrLS0tiV8oKSlpk6wbGxuzzdrghk6OhCEzfPhw/EATEhJCnjIHAADQ9aP1f/7559GjR2wSeffu3TalhREREcFTfrl0OnevMBg29uzZ8/Tp05iYGIIgdu7c6ejo2I38SAAA9GRZ//Tp08OHD1etWmVkZER2aHRC1q3k5GS00D5B7PxImP9jfWHha9euWVlZNTQ0NDc3z5s3Lzo6uh3PHAAAABjeOGGKioqMjIzGjBmjo6OjRaITnMU4uYqBgUE7ft4lkTBkTExMcLqoxMTEHTt2QKcEAKDrZd3IyCgnJ6epqamTj76yshKHwZDzLHYvVq1ahdPLnT59GqaeAgDQITcAT7YiKys7ePDgNWvWWFtb9+3bF38+btw4chZA/g3V2y3rXRUJw8aVK1csLCzKyspYLJarq2tiYiJbVjkAAIBOHa2Xlpbm5+dLSUklJyc/JdHS0sLXo8eyLi0t3b6pT10YCUNGXV3dy8sLLefn58PUUwAAuni0rqOjc+bMmc4/eizrurq63f1KTJ8+ffHixX///TeKApo8efLcuXOhgwIA0DWyjipgpKSkNDQ0sFislpYWGo1WWFg4a9asNvk3Wlpa7t27h5ZxiAs3st5ux3rXRsKwcebMmZcvX6KTWr16tb29PUw9BQCga2SdwWBs3LgxJydHWFiYSqXS6XQWi9W/f/9Zs2a1dTs+Pj7tGK3r6+u378gPHDggOBdDRkbGx8dn5MiRLS0tMPUUAICulPXk5OSSkhIvL6+3b982NDTMmDHjxo0bX1cI+yEiIiK40IR/K/werQsadnZ227dvP3z4MJ56unnzZuimAABwD29emZaXl5uamqqrq+vp6SUnJ4uLiy9atCgoKIhGo7VpOxQKBce8s5Wj/BoWi5Wdnd1BWe/M6khcsm/fPisrK7S8c+fO9+/fQzcFAKCzZV1NTa2srAy9uvz06ROTyaRSqRISEnzNt15UVIQj5dst6wISCcP2yHLt2jVUILi5uXnmzJko3TEAAEDnybqhoSGNRtu3b5+8vLyKisqFCxeuXbvGYrHYivrzlo4HrQss/fv3P3r0KFpOS0tbvHgx9FQAADpV1kVFRXfv3o3CNjZs2JCSkhIREYErRfBb1pWVlaWlpdu3EYGKhCGzdu1aZ2dntOzn53fy5EnorAAAcAPPFE1LS+vXX39FiQTOnj3bCYfOk/elAhUJw8aVK1cSExNTU1MJgnB3d7exsbG3t4cuCwBAJ8l6Q0NDTExMdnY2i8XCH86fP59/8/J7XhgMGzIyMvfu3bOxsamvr2cwGLNmzXr37p26ujr0WgAA+C7rDAZj+fLl0tLSffr06UgSGBaLlZaWhpbLy8s7QdYFJCfM9zA1NfX29p4zZw56RTx79uygoCDB9BoBANCjZD0+Pl5OTs7Dw6ODib3odPrWrVs7c7S+a9euNWvWCKysEwTh4uLy5s0bT09PgiBev37922+/4US+AAAA/JJ1RUXFpqamjs+HFBYWXrlyJVqOioqKjY3l8HyQl5fXcVnvFpw8eTImJubt27fo8WL48OG//PIL9F0AAPgo6/r6+kOHDj1x4oSdnZ2ioiL+3NjYuE1aT6VScebxqqoqDrKel5eHpxG1O3OAIEfCkBEREfn333+trKxQfdfFixebm5sbGxtD9wUAgF+yTqPRcnNzExISQkNDyZ+3tZYp92APDJVKJWd4byuCHAlDRlNT89atW2PHjmUymTU1NTNmzIiMjJSUlIQeDAAAX2Q9JSUlNTX1xIkThoaGfK2b8bWsa2pq8unOIWj8/PPPBw8e/O233wiCSEpKWrZs2bVr16AHAwDABm8kmMlk6urqGhsbd46m8zC6UQBzwnBg+/btU6dORcvXr1/HadEAAAB4LOvm5uYMBuPt27edJpG8knUBzAnDAQqFcvXqVUNDQ9TcuHFjZGQkdGIAAMjwxglTVFTU2Nh4+PBhKpUqIyODX5NevnxZREREkGW92yEnJ+fn5zds2LDGxkYajTZz5sy4uDhlZWXoygAA8FLWFRUVFyxY8LUHhn9BJryS9W4RCcOGpaXlhQsXXF1dUUTQ3Llznz592mnuLwAABP2xnjzXv91kZGTs3bv30qVL4uLiHdkOjUbD+a0Q3t7eqqqqbF9rbGyUkpJCR/769evemSll5cqVFy5cQMu7du3qLiE9AADwG94M8aqqqpqamjo+YBQSEhr9BQ7D8JycHHw36m1OGMyZM2esra3R8qFDh548eQK9GQAAno3W6+vrT5w4UVNTM2TIEHl5efz5uHHj2q31N1v55mg9ICBg4sSJKCFwY2NjR24nAp4ThjO5ublWVlYoeY6CgkJsbGyvvckBAIDhjVu5pKSkqKgIuUTIn48dO5YfPl/sWNfR0eng9gU/JwwH+vbte+PGjQkTJrS0tFRWVjo7O4eGhsIcJQAAWecBenp658+f77SDxrLekbQBPQNHR8e9e/f+/vvvBEHExcXNnz//7t278PoUAHozvPz/Lyoqevbs2d27d2NiYvhafpOH0Y3dMRLm6wcO5JIiCOL+/fvu7u7QrQEARus84MWLFxcuXKBQKBoaGr6+vjIyMvv37+fTaJqHst4DAkgoFMqNGzdGjBiRnJxMEMTx48eNjIyWLl0KnRsAYLTefkpLSy9cuLBy5cobN254enpeu3ZtzJgx/Cu/2WvnIn0POTk5f39//G551apVgYGBYBYAAFlvP2lpaX379h0zZgzKuiUjI+Pi4lJeXl5VVcXzI25ubq6srETLampqHdxa98oJwwFdXd3//vsPzRtgMBjOzs5o8A4AQG+DN04YMTGx6upqJpOJQ0rq6uqam5u5iTDJy8vDr1vpdDqqyIx5+PChlJQUWh47dqyysnJxcTFeGx0dXVBQgNLzzp49G39eU1Pj7++Pm6qqqg4ODriZmZmJU6ns2LFj1KhRQ4YMIW8zPT0dN0eNGqWhoYGb/v7+NTU1uDlr1ix8jjQazc/PD6+Sl5fHueMJgsjPzydnLTY0NMRR5wRBJCQkkFV4xIgROjo6uPn8+XNyCUAnJydsExQJipfXrVt3/PhxFotVXV09efLkJ0+evHv3Dq/t27evra0tbiYnJyckJOCmlZWViYkJboaGhubn5+Pm+PHjFRQUcPPu3bs4lw5n42toaIwaNeqbxkfZhCwsLHAzIiICP4oRBOHg4EAOb+Vg/Kampvv373NpfBMTEysrq/YZf/r06eQ5d2TjS0lJOTk54WZJSUlQUBCXxre2tsapfgiCePnyZWFhITfGFxERIc/g42z89PT06Oho3BwwYICpqSmXxn/48GF9fT03xldSUnJ0dOTS+HFxceR/eXt7ey0tLW6Mz2Qy79y5w6Xx9fT0hg0bhpuJiYlJSUlcGn/SpEmysrLcGL+ysvLp06e4qaWlRZ4pydn44eHhubm5uDlt2jQJCYkulnVTU1M6nX7u3DkHBwcDA4OkpKTHjx/369dPRkbmh7+l0+moOgS6VGxrq6uraTQaeS1Z1lksFhq5s90/UMAfbrIZiEaj4bUsFqu5uZm8tqGhgfxbBoNBXltTU0NeywZ5Fdsh0el08tqGhgby2sbGRvJatuxjdXV15LUtLS3f26m5ufmhQ4d27NiBXFULFiyYP38+TsvDljqmqamJ/Fs2O7DtlO3SVFdX4+9zNj5bHyAbHx0DB+Oz2YGD8dl2ytn4jY2N/DA+W1dh2ymfjC8mJsa98clPum01fnV19ffiINh2ynZIgmB8tsf6NhmfbaccjM9kMsk/JM/gaavx2XbaVngzHQn5YU6dOvX582fU7Nev3+bNmzviJPnedKTHjx9PmTIFzUViux7tYPfu3Xv27OlhRZ9//fXXv//+Gy3Pnj375s2bHa9HCABAr3DC1NXV7d+//+jRo7W1tQ0NDefPny8uLi4vL1dSUlJTU+OTlODR+tezT9tBj0ylcvHixezs7JCQEIIgbt++bWhoePDgQejrANBL6NArUwaD8fHjx9zc3KKiogcPHlAoFHV1dTMzM3V1df4ND7Gsd/x9aU9FRETEz8+vX79+qHno0KErV66AWQAARus/Rl5evn///mvXrhUXF6fT6S4uLmxf8PX15Xm+dZSlgCAIdXX1jm+tW+eE4YCCgoK/v//QoUPLysoIgli2bJmiouK0adOgxwMAyPoPOHjwYEZGRlpa2suXL+fPn8++dT74rHk7Wu/WOWE4o6+v/99//zk6OtbX1zOZTBcXl4CAgJ9//hk6PQCArP/geb9fv37y8vIFBQUWFhZ8qoXEP1nv2YwYMcLPz2/KlCl0Or25uXnq1KkhISGDBw8GywBAD4Y305GysrJCQkI6J8MUb2V97dq1PSwMho1x48b5+PigS1NbWzt+/Hi2mQEAAMBo/RsYGxvLyclFRkYOHTq0Iw4NJpN56tQptJyTk9MJsn78+PEef43RjN81a9YQBFFWVjZ27Njw8HBtbW3o/QAAsv5d6uvrRUREjhw5IiQkRJ4B0dYS1S0tLREREVjiv/4CeTILT16Z9hJWr15dVla2d+9eNK137NixYWFhUNgaAEDWv4uCgsKcOXO+sfU2+jdQZB5aRtORvjdU59Vo/ezZsytXruyRr0zZ2LNnT1lZ2dmzZwmCSE1NHT9+fEhICDfTgAEA6I2yLiMjg7If1NTU1NXVqaqqCgkJ8SN0neeyvmXLFjc3t94g6wRBeHh4lJeXo5tlbGzs1KlTAwIC2CZAAwDQ3eHZS87MzMxNmzbNnz/fy8srNTV1x44dKAkXn2RdRESEnPwI4AYKhXL16tXx48ejZkhIiIuLS89IYAkAAI9lvaGh4eDBg4aGhqh6g56enqqqKnre55Osq6qq8uRpoMdHwrCB3Fw4m92DBw+WLFnSwbxCAAD0QFlPTk6mUqnLly/X1NQkCEJSUnLdunW5ublseQp5KOu8el96/PjxXiXr6Or4+/ubmZmh5tWrV93c3EDZAQBk/f9Ap9NlZWXJHurm5mY6nc6WfJKHsg5zkTqCoqLis2fPcEnCv//+G5QdAEDW/w/9+/fPz89/+vQpynpcXV3t4+Oj2IqAy/rZs2d7p3NZU1Pz1atXuHoAKDsA9Bh443+Ql5dft26dp6cng8EQEhJatGiRtLR0O0rgs1gsXM2HXKAEg/N88UrWe1UkDBtaWlohISE///wzqgaFUrR7e3t3zmxhAAAEWtYJgrCzszM3N3///n1ZWZmamtqgQYMkJSXbuhE6nY6mzHTaaL2XA8oOACDrnEBDdSEhIWlp6fbpgoiIyNatW9FyeHj4mzdv+C3rvS0SBpQdAEDWuSUwMBBVmlZWVi4sLFRTU/v999/bmniEQqHgoq75+flssk6uT8jDSBjoBKDsAACyzk51dfXZs2e3bt1qbW0tKipaUVFx//79I0eOnDt3jofHWlJSgiuvghMGlB0AgG/Cm3/apKSkQYMG2draioqKovi5BQsWVFRUlJeX8/BYeZ45oDdHwnxP2SE2BgBA1v9/+vTpk5aWVl9fjz8pLCxkMpmysrL8kHURERFehU5u2bKFTqdDP/iesi9evBgFrQIA0F3gjRNGTU3N0NBw48aNQ4cONTQ0TE1NjYyMNDU1DQwMJAhCVlbW1ta243vB0Y0qKir8K4ENyk72xvj4+JSXl//7778SEhJgHADoRaP10tLSqqoqCQmJ9+/f37t378OHD9LS0pWVlU9bCQ8P5+1onYeZ1iES5odjdn9//7Fjx1ZVVYFlAKAXjdZ1dHTOnDnDk01hTw6NRvuerPPwfSlEwnxP2V+/fj1+/Pj379+jYNOffvrp2bNnGhoaYBwA6BWyzitoNNo3y3HwT9aB76GhofH69WsnJ6fXr18TBJGYmGhra/v8+XM8igcAQDARrPA1ISGhKV8wMTHpBFmHSBgOyMnJPXv2bOrUqaiZlZVlZ2eHszsAAACyzpWsL/2ClZVVJ8g6RMJwRlxc3M/Pb8mSJfgSjBo16tWrV2AZAOjhss5kMrOysvh9rDgSBopTd/K91tvbGyduq6mpGT9+/IMHD8AyANCTZf39+/dbtmzheXZ1MgwGo6KiguejdYiE4ZLDhw+fPn0axZU2NTU5OztfvnwZzAIAPVbW5eTkhISEqqur+XegfMoc0AurI7WbDRs2+Pr6ioiIoOczNze3o0ePglkAQNDgjaLJysqamJisW7fOwMCAXDl606ZNvEplzo/MAUBbmTdvnqKiorOzMypn6O7uXlBQcOrUKbg1AkBPG60zmUwNDY2RI0dqaWlJkeDhgWIPDIVC4WHRJYiEaSsTJkwICgrCl8DT03PcuHH46gAA0OVQsGdD0LjZire3t6qqKkEQd+/enTlzJqrEhNPzdhxxcfGqqipxcXHoCm0iJSXF0dExPz8fNQ0NDR89etSvXz+wDAB0OTx7dm5oaLh9+3ZsbGyfPn1cXFzevHkze/Zs5IflHgaDsXnzZrTMNlsdSznZyQN0Ff3793/79u3UqVPj4uIIgkhPTx82bNidO3ccHR3BOADQtfDGCdPS0rJv377ExERjY2MajSYlJfXmzZvTp0/z8ED5JOsQCdNutLS0QkNDnZ2dUbO6unrixImenp5gGQDoCbKekpKSnZ29f/9+Ozs79Erz+PHjCQkJbXVbCwsLn/nC+PHjO0HWIRKmI0hKSt65c2fPnj0o8JHJZK5bt27FihWQyxcAur2sl5eXGxkZSUtL408kJCSoVGpZWZmAj9aBDkKhUPbu3Xvr1i2cuffixYuOjo7wEhUAuresGxgYfPjwIS8vD3/y8uVLOp2O3nYKsqxDJAxPmDVrVmhoqKamJmqGhIQMHTr048ePYBkA6K6yrqmpOWXKlDVr1vzzzz+5ublr1679888/Fy1axMNiF3ySdcgJwysGDx4cHR1tY2ODmugl6rNnz8AyANAtZZ0giF9//XX37t0mJiY6OjpmZmZHjx5lc453EPxQD04YgUVDQ+PVq1dz585Fzerq6kmTJnl4eIBlAKAz4eXbwiGt8OlAIRKmWyAuLn79+nUzM7Ndu3axWCwmk7l+/foPHz54enqi8uUAAAi6rLu6us6dO9fQ0HDfvn1fr718+XJbQ9c7WdahOhI/2LFjh6mp6YIFC+rq6giC8PLyiouLu3Pnjp6eHhgHAPhNh2aZslis8PBwQ0NDaWnppKSkr78wdOjQH7rXi4uL/fz80DKNRgsJCSFvf/Xq1XJyciwWy87OrqWlhSCI58+fjx07liCIuLg4XBiPSqViry5KMRgfH4+bsrKypqamuFlaWpqRkYGbWq3gZlZWFjn/TP/+/eXk5HAzISGhsbHxmyfIZDKjo6PxKikpKQsLC/JtKTU1FTfV1NTIGpeXl1dQUICbRkZGSkpKuPnhw4fa2lrcHDRokJiYGG5GRETgZTExsUGDBuFmXV0d+booKyuTaxt9/vw5NzcXN3V1dckZj9PS0sjRLJaWlpKSkrgZExODoxi/Z/z09PStW7fidMpycnJXrlyxt7fnYPyMjIzS0lLcNDU1lZWV5cb4dDo9NjaWS+NraGjo6Oi0z/iDBw8mj1Q4GL+mpiY5OZlL4+vr65PjCz5+/EiejsfB+MLCwuRHZLaeLy8vT576W1JSkpmZiZt9+/bt06cPl8Z/9+5dc3MzN8aXkZExMzPj0vg5OTmFhYW4aWJiQh63cTA+i8WKjIzk0vgqKioGBga4md8Kl8YfOHAgeRY6B+M3NDSgIpEIRUVFY2NjLo2fnp5Ojhu0srLqyNNth0brVVVVJ0+evHfvXnFxcVFR0bRp09q3kadPn35v7YsXL0RFRZuampCmk0frz549w5dcWFiYrCy1tbUPHz7ETT09PbKs5+Xl4bWxsbFbtmwhK0tCQkJUVBT52pBlPTg4mNz1hwwZgn04dDqdvFMNDQ2yshQWFpLX2tjYkGU9OTmZXJvCxcWFrCyvX78mRxmZmJiQZZ28WQUFBXLnrqysJK+1tLQkK8unT5/IrzQnTZpElvWoqCjyP4aOjg5ZWfz9/fE/OQfju7i4BAcHo75eXV09Y8YMFxeXvn374gRwDg4OZOO/e/cOTVtFqKqqkpWFg/Gbmpq4N76trS1ZWdpkfDMzM7KsczB+WVkZ98afNm0aWVkiIiLS0tK4Mb6YmBhZWdh6vrGxMVnWs7OzyWvHjRtHVhbOxg8MDCQn7eBgfG1tbbKsczZ+UlISuYT9/PnzybLOwfhMJpN741tZWZFlPTU1NSgoiEvjGxgYkGWdg/Grq6vJOzU1NSXLOmfjx8TEkG8JZmZmHZH1Do3WGxoa5syZs2zZMklJyYCAgBUrVrB9QU9P74ejdRaLRY5FweOmJ0+ePHv2DOWEycrK0tfXx7c18hXquC8YcsLwFRaLdfz48Z07d+IxzuDBg+/cuYMvKAAAAuRbl5SUdHJyunLlCtLlDRs2sH3h7t27P7znUCgU8nfwGJacppE8TOBh+kagM9x8FMq2bdtsbW1dXFzQk29sbKyVldXly5dnzJgB9gEAwZJ1giCWLFny66+/Jicn3759e9euXWxreRX8gGWdQqGQXSIdByJhOgdbW9t3794tXLgwICAAPa46OzuvWbPm5MmTECEDALyFypPhmImJyaZNm0S/gldHiWVdTk6OSuVlWW3ICdNpKCsr+/v7Hz58GDvWz549O2LECPJ7JAAAun603jkBjpAQpmdAoVDc3d1tbW3nzJmDXqIgh4y3tzfOBAkAQBfL+qpVq3R0dKSlpVetWvWNrfNoIMw/WT979uzKlSt5VZkP4AZ7e/v4+Pj58+ejUJDq6uqZM2euWbPmxIkT5AgfAADaR0cdGkOHDlVXV5eWlh5GwsjIyNLSctiwYbzKCcM/WYecMF2CsrJyQEDAoUOHyA4ZW1tbcoAzAABdI+uYFy9e/PXXX6hY3eLFixcsWBAaGtrWjbS0tAR9ISsrqxNkHegqKBTKjh07goODcfRubGzswIEDT506hecoAADQDnjjJCkuLvby8lq2bFlzc7Ofn9/ChQs1NDQ8PT3t7OzaNGBnMBhnzpzpzNE6RMJ0LT/99BNyyDx//hxNbNm8efP9+/f//vtv8uQdAAA6e7Senp7er1+/sWPHJiQksFisSZMm2draiouLl5SUtGk7IiIie78watSoTpB1iITpclRUVAICAk6cOIEnhYWFhQ0YMMDT01Ng66cDQM+XdSqV2tTURBBEdHS0paWlhIREXV1dc3OzsrJyWx/Mrb6goaHRCbIOCEQvpFI3b9787t07nISgoaFh3bp1o0ePJvviAADoPFm3tLQsLCz8448/goKCRo8eXVRUtGPHDgsLC15FmPA1EgaqIwkI/fr1e/PmzR9//IFnPLx8+dLS0vLChQtgHADobFmXkpL6448/1NXV169fb29v39LS8tNPP23ZsoVXRwmRML0EISGh3377DQWzo0/q6upWrlzp6OhITvkEAADfZR3lmfz1119HjhyJEptNnDiRV7NMWSwWzpMJCWF6A+bm5pGRkXv37sVz2V68eGFubn758mUwDgB0nqzzJMDxm9TU1HydlZdXQCSMYCIsLLxnz56oqChLS0vcDdzc3CZOnEjOjQ4AAL9kHQU4GhkZ4QDHTZs2nTt3jieRDOT0jRAJ06sYOHBgdHT0zp078UuagIAAc3NzX19fMA4A8FfWeRXgyGKxSr5QX1/fCbIOCDiioqIHDx58+/YtLoRSVVW1cOHCyZMnk6ssAQDAY1nnVYAjnU53+wKuJMK/rLwEQXh7e0MkjOBjbW0dFxe3detWnL/T39/fzMxsz5495DJ4AAAIXICjkJDQnC+Ym5uzyTrPs/ISBLFmzRqIhOkWiImJHTt2LCwsDFdxa25u3r9/v6mpKbmWGAAAghXgSJZ1XIsS5iIBmOHDh79///7IkSNSUlLok+zs7KlTp06aNAl8MgDAS1lnC3Ds06ePs7Mzz0sj8UPWly5dCq9MuxciIiLbt29PSUkh18x78uSJmZnZ77//Dj4ZAOCZouXm5oaFhTU0NLBYrJaWFjqd/vnz53379nW8jEZFRQX/ZN3T0xM6QXdEW1v77t27z58/X7t2LaoT39zcfODAAV9f3z///HPq1KlgIgBG6x2ipqbG3d09ODg4JiYmMTHx06dPQUFB2traPBkIgxMG+B6Ojo6JiYkHDx6UlJREn2RnZ0+bNm3SpEnp6elgHwBkvf2kpaWJiYldunRp6tSpFhYWJ06c2Lp1a0lJCU/KaPBV1iESprsjKiq6c+fO5OTkadOm4Q+fPHlibm4OPhkAZL391NfXGxgYUCgUPT29lJQUgiBGjBiRnZ1dV1fHk0cBtMDz6EaIhOkx6Ojo3L9/39/f38DAAH2CfDKmpqb//fcf2AcAWW8z2traGRkZjY2NOjo6WVlZyMNOb6XjG6+trUULsrKycMEADkycOPHDhw979+7FqduRT2bMmDExMTFgHwBkvQ3o6+traWmtXr1aQkLC0tJyRyvKysptdZvQ6fSFX7h//z6brMvIyPD8/CESpochJia2Z8+e5OTkyZMn4w+DgoKsra1nzpyJXq4CAMg6V+zevXvZsmUo1W3//v11dXW3bdvW5qOhUvW/gG8JfJV1T09PkPWeh56e3qNHjx4+fKivr48/vHv3rpmZ2fLlyz9//gwmAnowFIGtK3azFW9vbzMzs7KyMoIgbt++PWvWLLhmQJue/7y8vA4cOFBcXIw/lJCQWLdunbu7u7y8PJgI6Hl0dKDq5ubGYe358+c7HreOX5nyY7Tu7e29ePFiXlVxAgQNERGR1atXL1q06PTp08ePH0d9qbGx8ejRo15eXu7u7mvXrpWQkABDATBa/39wTqpuZ2fX7hhHNFo/f/68lpYW3pednR1vz19cXLyqqgq/YQN6MOXl5X/88ce5c+eam5vxh5qamnv27Fm8eDH44gCQ9e/CYrF4Eq6OZP348eM4tVN8fPyAAQNA1oGOkJeXt2fPHh8fH/J8BRMTk4MHDzo7O4N9gB4AD16Zvnv37uTJk7gZGxu7YcOG6OhonhwfOfKdHwGOEAnT29DW1r5y5UpiYiJ5+lJqaurMmTNtbGyCg4PBREBvl/W3b9/u27evtLQUj/pVVVWFhYUPHjwYHh7OW1mHSBiAV/Tv3//+/ftv374dNWoU/jA6OtrBwWHcuHGxsbFgIqD3yvo///wzadKkw4cPY8dL3759jx8/PnnyZG9v7457ePgt60BvZtiwYSEhIQEBAQMHDsQfPn/+fMiQIY6OjjByB3qjrNfU1BQWFk6cOJHNmU6hUMaPH1/eSps2yGQyj38BDfaxrIuIiIiJifH8/CEnDDB+/Pi4uLgbN27gxAOo5LqDg4O1tbWfnx+ukA4APV/WhYWFqVQqLmhARkVFhUqltjV5QEtLS8QX8vPz+T0XCXLCAHggMmfOnJSUlHPnzmloaODPY2JinJ2d+/fv7+3tTaPRwFBAz5d1SUlJXV3dbzoiExMThYSElJSU2rRBERERvy/Mnj2bPFoHDwzAb0RERFatWpWVleXl5WVoaIg/T0tLW7p0qZ6e3okTJ/A4AwB6pqwTBPHTTz/5+PjEx8eTP0xOTvb09HRwcOh4gaT6+nq+yjpEwgBsiImJLV26NDU19c6dO1ZWVvjzz58/b926tW/fvjt37iwpKQFDAYL79Nnxt5o3bty4ffu2gYGBtrY2g8HIzc3Ny8ubNGnSkiVLOlJRGsWtm5mZHTlyBGX65UloDQC0iRcvXhw5coTt9am4uPjixYu3bNlCzjkDAD1ktE4QxNy5cw8fPjxs2LCampqmpqYRI0bs27dv6dKlHdF0DDhhgK5l7NixQUFBUVFRv/zyC+7STU1N58+fNzY2njNnDtujKgB0ObzxP5i2wo/j47esQ04YgBtQSExqauqxY8euXbuGXp8ymcxbrYwfP3779u3kEHgA6N6jdb7Cb1mHSBiAe0xMTC5fvpyZmblp0yZpaWn8+dOnT3/++WcbG5t//vkHyuwBIOs/gN8BjgDQVjQ1NU+ePJmbm7t//35lZWX8eXR09OLFi/v06bN+/XpU+hEAuoSuDwKprKzEaSBpNNqtW7fQMpolVFFRgZosFotGo5FDawoLC/FMIgqFoqmpiVcxGIyioiLcFBMTU1FRwc2Ghga82Tlz5jQ0NJBTfVVVVZGntiorK5PXFhcXk0f3mpqaeCoWi8UqKCjAq0RFRVVVVXGzqakJZY1HSEtLk5N9V1dXkyPnlJSUyNliS0pKyEHTGhoaZK8RCvD/3+UUFlZXV8dNGo1GjtmQlJRUVFQk3zKrq6txU15enjwCLS8vJw881dTUyDmWP3/+jCfpcDa+uLg4WfvIxkd5fsipfiorK3HsE5r9QJ6DxsH4LS0t5OIYnI0vIyNDrovbJuP36dMHedgVFBR2797t4uJy69YtLy8vfBWqqqo8WrGzs5s9e/bEiRNRp+VsfAUFBfL8j7KysqamJm6MT6VS+/Tpw6Xx6+vrccF3VByYPFribPyioiIGg8Fv47P9u33P+F//u7H1/Obm5tLSUtyUkpIiV2qraYVL46urq5OD5TgYn06ns6X1J0d4czZ+RUVFQ0PD9/7H20rXl9FITU3dunXr99aGh4cjFbC0tPTz8yNHE69cuRKbSVRU1MfHh2z6TZs24Wb//v337NmDm69evTp//jxuOreCm5cvX37x4gVu/vbbb+S0kZs3byZ3Jl9fX/z/1tDQ8Ouvv+JVurq6KIYHERMTc+LECdwcO3bskiVLcPPmzZvkSsrr168fPnw4bu7evfvTp0+46enpSb5Lubi4kP8VPT09cTM9PX3Xrl24OWLEiHXr1uHmw4cPb9y4gZuurq4TJkzAzZMnT5LztR09elRHRwc3lyxZgiWAs/EHDhzo7u7OpfHPnz//6tUr3Pz999/J72w4GL+6unr58uVcGn/ixIkLFy5sn/EvXrxIViVkfBaLlZ+fn5ubW1BQwPYPJSYmpqenZ2Bg4OjoyMH4bm5uY8aMwc0jR46Q38RyML6UlNTly5e5NH5gYKC3tzduzp0718nJiUvjr127liyUHIxvZGR04MABLo3v4+Pz5MkT3NyyZcuQIUO4MT6dTl+wYMH3en5ycvL+/ftxc+TIkStXrsTNu61wafxTp06RtZuD8XNycrZv346b1tbWmzdv5tL4Hh4eb968wc0LFy50pMZL18t6c3Mz/l9lMBg3b97EHbSwsDAlJSU9PZ0giL/++ot8YQBAAPn06dPFixf/+ecftrQZFArFwcFhxYoVU6dOhXkSAL8R9KJ30dHR6KHS19d3/vz5PN8LRMIA/Bip3L1798KFC2FhYWyr1NXVlyxZsnTpUvLoGwB4C0TCQCQMwGPExMTmzZsXGhqalJS0du1a8tN0UVHRoUOH9PX1J02a9OjRI0gzB/RGWed38gAA4B9mZmYeHh4FBQVXrlyxsbHBn7e0tDx58sTJyUlPT2/fvn3k940A0MNlncFgYB8R5IQBuimSkpKLFy+OjIx89+7d8uXLyeFGeXl5e/fu1dbWdnBw8Pb2JscIAUCPlXW8zCdZh+pIQKcxcODACxcufP78+cKFC+TCHS0tLcHBwUuXLlVXV588efL169fJIbYAALIOAAKNjIzM8uXL3717FxERsXjxYnKMPJ1O9/f3nz9/vqqq6qxZs+7du0cOoAaAbinrLBYr8wuVlZVkWedHfWqojgR0IUOHDr1y5UphYeHff/89btw48lNjY2Pjv//+O2PGDDU1NVdX16dPn5L/FwCAM4IV4Eij0ciTU8rKyt6+fYvCfhkMBk9SQrIhLi5eVVVFntgGAF1CaWnp3bt3b968GRYW9vV/pbKysrOzs4uLi729PT/+EQCQdX7R0tLy+PFjtBwXF/fkyRM00VFaWppPVWlA1gFBIz8///bt27du3YqJifl6raam5qxZs1xcXMihNQAguLJO5ubNmydOnIiLi0MZEvgUBLZ27drTp0/DW1NAAElPT0eJfz98+PD1Wn19fRcXlzlz5pibm4OtgG4j64cPH05MTCQIwtjYODU1Fa4W0DtJTExE+p6Zmfn1WgsLi8mTJ0+cOHH48OEwXxroNpEwEAYD9GYsLCwOHTqUkZERGRm5ceNGcuYpJPqHDx+2t7dXUVFxcXHx8fGBUqsg671a1iESBuhG2NjYnDp1Ki8v7+XLl8uXLycn3UWZdW/fvu3q6qqurm5jY7N3796oqCicRRboPQi0E2bXrl3oqdPJyYmcOpWHwCtToPvCYDBevnzp7+//5MmTtLS0b35HRUVl/PjxEydOHDduHDnnOACj9R47WgeA7ouwsPCYMWNOnz6dmpqanp5+5syZcePGkWtfoLhJX1/fOXPmqKio2NnZ/fHHH1BTG2S9h8s65IQBegYGBgbr1q17+vRpRUXFw4cPV6xY0bdvX/IXmExmeHj4zp07Bw0apKmp6ebmdu/ePT7FDQNdi2A5Yeh0Oq7109LS8ubNG/TyZ+vWrceOHYOrBQBtIikp6Ukr4eHh35ynKiIiYmdnN2HCBHt7+8GDB5NL6wHd+DFOsJ4dqNRhw4ah5dzcXHDCAEBHMG9l27Zt1dXVz58/f/LkSUBAALneJp1OD2kF1d60sbGxt7e3s7MbPnw4n9J1AL1utE7m5s2bK1asQJVkT58+vWHDBn7sBaojAb0KFosVGxuL3rLGxMR8L05GSEjI0tLSrhV7e3sNDQ0wHcg6b2Tdzc0NVeP29vYmF3TmIRAJA/RaSktLnz59GhQUFBYWlpGRweGb+vr6SN/t7Oz69esHphNwBPptIXbCwPMgAPAcFRWVBa0QBFFYWBgWFhbayvv379lG8Sipqo+PD0o6ZvcFKysrcMeDrLdT1iESBgD4ioaGxsxWCIKoqal58+ZNaGhoWFhYVFQUW873srKyB62gwk9Dhw5FA/nhw4eTCz8BXYjgOmF8fX0XLlyIlsPCwmxtbeFqAUAnQ6PRoqOj0UA+PDy8qqrqe9+kUqnGxsYDWxk0aNDAgQNVVVXBgCDr/4dLly4tW7YMLSckJFhaWsLVAoAupKWl5cOHD6FfKCgo+OETwEAShoaGkCm+t8u6h4fH+vXr0XJmZqaenh4/9gKRMADQPrKzs5G+h4WFpaSk/PD70tLSlpaWWOUtLCwgVKFXyDqTyTxw4ABa/vDhw927d9FyaWkpW1YjXgGRMADQccrKyt68eZOQkBDfSlZW1g+FRUhIyMTEBHtsBg4cyKf/cZD1LoZOp+NAxvz8fDRLgiCIpqYmtkwXIOsAILDU1NS8f/8+Pj4eCX1SUhI3tbY1NTWRvg8YMMDIyMjAwADmIfYEWSfj7u5+9OhRgiBERUWbm5v5tBeojgQAnfAUnpqaigbySOi5zAivqqpq+AUDAwO0oKioCCbtrrK+YcOGM2fOEAShpKRUVlYGlwoAegxFRUVklU9LS+M+L7y8vPzXWq+urg5W7Qayvnz5ci8vL4IgdHV1s7Ky4FIBQE+loaEhKSkJC31CQkJ9fX2btiAlJYUlHi9oaWn1ztgbwXU+YGccX/1rvr6+c+fOhUgYAOhCJCUlbVrBn3z+/DkjIyM9PZ38l0PUfH19/ftWyB+KiYnp6ekZGhrq6empq6traGiof0FVVbUH/9cL7mh91qxZ//77L0EQI0aMCA8P59Ne4JUpAHQXKioq2IQ+IyOjqKioHZuiUqkqKipI4slyj5e7dcISwR2tNzY2dsJoHQCA7oKioiLboB6N078e1+fl5XF21re0tBS3kpCQ8L0HCHUSbNKvqqoqyMlwersTZsGCBeCBAYDui5SUlGUr5A9pNFpWVhZZ6AtbKS0t5bIkfUNDA0pw9m0vB4WipKSkqKgo8wVZWdkfLsvIyIiKivY6WWcymXfu3EHLhYWFaIGvT0OXLl2CfwwA6GGIioqatPL1IL20tLSIRGFhIblZXV3NzfZZLFZZK209MDExMW7uBFOnTpWUlOw5sn7z5k20XF5eDk4YAAB4CJVKVWtlwIAB33MSfK31+AZQXFxMo9E6cgDNrfzwfpCfn99zZF1ERMTb2xstjxw5Er0MgUgYAAA6B3Fxcd1WvveFiooKpPLFxcVVVVW1JGpqar5u0un09j1tdOQsBEvWKRQKTuaJ74p8lfWlS5fOnDkTZB0AAG5QbMXU1JT74fn3FL+mpqauru6bazuYKwUiYQAAAPiFWCudnMWsk6ZglZaWpqWlcfkOujNlHSJhAADoYfB9tE6j0fbu3Zueni4iIiImJrZ//34tLS1uftg5AY4QCQMAQA+D76P1mzdvMhiMixcvXrp0acCAAX/++Sc3v2KxWDhrI9SnBgAAECBZf/78+eTJkxUUFCQlJefPn5+WlpaXl/fDX9XV1eGsBvyOhGmTawgAAEDA4a8TprGxsba2VkdHBzWVlJQkJSULCwu1tbXxdxoaGj5+/Mj2w9LSUrycl5fHvzRsbm5uRkZGnTP1CwAAgBssLCw6kpyAv7KOkuXLy8vjTyQlJfE8I6zae/fuZfshnU43NDRkMBh0Ov3ixYv8k10Gg3Hw4EGonAsAgOAwd+5cFxcXAZX1r2GxWGx3IXV1dVyKmkxERERkZOTs2bOxE0ZCQoKsv42NjTibD4VCkZSUfPXqVXx8/NKlSyUkJBoaGvA3hYSEyDkaGQwGdtzn5OQQBLFmzRocD0Oj0cgzCMTFxcmhMo2NjZGRkQkJCfPmzZOQkJCSkiKf2vd2ig5s4cKF+P4kIiJCvlfR6XTy7DUxMbFz584RBIEs09TURPYUSUpKUigU3MSZqSMiIpKSklasWIGDXltaWnBA0Td3Ghwc/PHjR1SkW0xMjFwlqrm5mcFg4CY2fkBAQEZGxpIlS9iM/z07CAsLh4eHJyYmogMjGx9NuyD3B2x8NCuNfF3YrjhKBkLe6X///VdYWLhw4cKvrziTySQXXcN2CA4OTkxMXL58Odm8bHZgM/7r168zMjLWrFnDZnw0g1FCQuLrnYaFhX38+HHZsmXktV9fcfJOUQEZNze3H/Z8vNNHjx4VFBQsX76czfjkCGg244eGhqampmIj/7Dnt7S0BAcH5+TkLF68mMueTxDE/fv3S0tL582b981O+M2dotNfv349lz2fIIjAwMDCwsIVK1ZwvuJk4/v7+9fW1rq4uHDu+eSdos6/ePHib15xtn83ZPwHDx6UlpYuXbqUy56POj/3RUW6QNYVFBRQYUM5OTl8JdTU1MjfkZOTc3Bw+OZIPzIycuzYsXiC0g/JzMyMj4+3s7ND++WG/Pz8p0+fjho1ivsHgvLy8oSEBEdHR+73gg5swoQJ5H8GziBZ/6ZlvkdxcXFSUtLIkSO530tmZubHjx8nTZrE/enHxcVlZWVNnjyZ+wPLzs5OTExs04EhWXd0dOR+L5GRkaWlpU5OTtz/JDU1NTExcezYsdyffmpqalZWVpuuS35+/sePHx0dHbnfC9K1Np1LREREUVHRmDFjuP9JZmZmampqmzp/ampqXl5emw4sNDS0pqamTT9Bp98mIycmJpaUlLTpJxEREUwms00Hhjp/W0+/urq6TQeGZ9q3G/46H1DamoyMDNT8/PlzU1OThoYGPGQBAADwCf6O1ikUyoQJE/79998+ffrIy8tfunTJzs6OfxOu+vTpY25u3qZ602lpaWZmZm3yraurq7d1L+jA2jTvSUhIqK0hOu3YC/pJm05fW1u7traW3wfWjtPX0dEhPzsL2um3aS+dc/paWlqdcPq6urptnQrfOaevp6fH/QN3Z55+x1/18b06UmNjo4eHx5s3b0RFRU1NTTdu3Eh+g8qBm614e3tz74RpBwJbHenPP/8MDg5++PBh7xxuwOnD6ffa0z9+/LimpubcuXMFdLSO3vZs3769vr6+paUFsrsAAAB0bycMhvt3ZZ0M5IQBAABkvZOgUqmdUC1QYHPCCAsLC3KxRDh9OH04ff6dfgfHmnz3rQMAAACdOibu5ecPOWEAAOhh9PbRusBGwgAAAMBoHWg/LS0ttFbodPrXd3pmK512MCwWq4OFgL93/B05kerq6oqKCn6cL4PB6OBkcW4skJOT83XMdWNjY3Z2NnlSe5dTXl7OZfl/AGT920AkDOLOnTvOrcyYMWP69Omurq4+Pj440Yevr++xY8c67WDy8/OdnZ3J6TU6yMGDB2/dukUQxN9//33q1CmUFeTmzZttuu0dOHCAnISEh7i7u9+/f59/9szLy5s/f/6GDRseP35M/vz169dz5szZvHlzcnJymzZYV1fn5+fHp6Otra09dOgQvPPrCMK9/PyhOhJGU1Nz586dKG9PXl7epUuXCgsLt2/fThCEnZ0dD0X2hygpKa1fv54faTtHjhyJRqYZGRm3bt2aM2cOlz98/Phx3759yQmluxFv3rxRVFQ8ffo0m0mDgoJGjRq1fv16cv4sboiMjAwKCpoxYwY/jlZXV1dNTe3JkyeTJk2C/0oYrQMdQkRERKsVExOTMWPG7Nu3Lzw8PCoqCs0pQzMPSktLMzMz6+rqwsPD4+PjGQwGk8l89+5dVFQUeep2Y2NjVFRUUFBQQUEB/jAhIaG5uTktLS0wMDArK4v85fDw8MDAwPT09P91SipVQUEBa01xcXFwcPDbt2+xD6G2tjY5OZlGo8XGxr58+bKyshJvjUajxcfHP3v2LCoqipxWEIFOhEajpaamosxNpaWlcXFxZJ9PYmJiVVUVmwfjwYMH48ePR824uDgGg/Hx48fAwMDc3Fw8gI2PjycPkFFyUGyxsLCwhIQEJpPJYDDi4uLYLIYSb7FZhiCIioqK0FbwOTY0NLx//766uvrly5dlZWVf+3NiYmICAwMzMzPRJzk5OdnZ2dLS0klJSWQH1IcPH0pLS0VFRVNSUvAxBwUFRUVFkXMfMpnM9+/fP3v2DNu/pqYmJyenqakpLi6OyWR++vSpqKgI7x0bMyEhoba2NjQ0FJ9RVlZWYGBgbGwsOWVjcXFxUFBQaGgoucTC+PHj79+/z2/HVA9G6Otc54JDS0tLW8cRbcXX17etaTF6JElJSXl5eRMmTMCfKCsrP3v2TFJSctCgQffu3YuIiLC3t3/58uXNmzefPXtWXFx8//793NzcoKCg6Ojo2NjYe/fuOTk5USiUgoKC7du3x8bGlpWVXb16VUJCol+/fgRBrFu3LjMz09/fv6qqysfHR0ZGxtjYuLy8fO3atbm5uVVVVTdu3KitrR00aFBhYeG2bdumT58uIiLy+PHjAwcOVFZWRkRE/Pfff/369VNRUUlNTT116lRYWFhKSkpycvLt27etra3l5eXLysq2bduWkJDQ0NDw5MmTp0+fOjg4iIqKvnr1SlFR0dLS8s6dO3FxcZaWlrdv3y4pKWlsbOzXr9/evXs1NTVRsZeioqItW7ZMnTqVnHA1Pj4+Kipq8eLFFAqFxWKtWLEiOzv72bNnFRUVV69eVVFR0dfXT0tL27dv3+zZs9FPbty4kZSUNGzYsJCQkFu3bj179qykpMTPzw9JZ0xMTGxs7P3795HFnj9//unTJ39//7q6uqtXr6IqCgRBxMTE7Nixo6Cg4MOHD7dv3zYwMNDQ0MjJydm3b9+7d+9CQ0MZDMbAgQPxcZaVlW3atOn169f19fX//PNPVVWVtbX169evo6KiqqqqKisrra2tcTz4v//+++nTp/r6+qampiFDhvj5+R09erSxsTEoKOjFixfW1tbS0tK1tbXu7u4RERHNzc3Pnz9/9OjRTz/9VFFR8ejRo/Ly8tra2mHDhp05c4ZOp5uamiLFX79+vaOjo5SU1Lp169LS0l68eJGSkuLo6Ojj43PmzBkajRYQEPDq1avhw4dLSEhERET8/vvvTU1Nnz598vHx0dfX79OnD0EQKioq9+7dMzQ0ZMv2CnRvJ8zLly8fPnyYlZVlYmKyceNG/l3dpUuXzpw5E9zr38TExASPRjFZWVknT540NDSMioo6ePDgjBkzXF1daTTawoULU1JSzMzMzp8/r6+v7+7uTqFQkpKSdu/ebW1tjf5dm5ubL126RKFQrl+//vjx40mTJkVFRUlLS584cQKN754/f052qhYUFFy+fHnt2rWjR49msVjnzp07ffr0hQsXkIRNnz59ypQpLBZr48aNISEhixYtCgsLU1JSOnDgAIVCaWxsnD9/fmxs7MiRI9lOQV5e3tXVdfv27cjpZG9v/+rVK3t7e4IgQkJCBg4cyJYBKjs7W0dHhzzCoFKpFy9eRE68J0+ecE67mpmZefr0aX19/cjIyEOHDjk7Oy9cuJBGoy1YsODjx4/9+/dHlvH29paSkoqKijp06JCdnZ2KioqHh8fMmTOdnZ0Jgrh3756np6eXlxcasI8cOdLJyYnt9e+FCxcUFRU9PT3FxMQyMzM3bNgwcOBAJyen8vLy4uJid3d38pdXr16dnZ1tb2/v5OSUlZXl4+Nz4MABS0tLJpN56tQpb2/vXbt2RUVFUanUs2fPotxbCxcufPv27ZRW/Pz8kPU4ICEhce3atZaWlqSkJD8/P2QEGo32xx9/+Pj4rF+//sWLFxMnTly0aBF6u5OXlzd48GCUIlBPTy87O9vS0hL+DXuIEyYnJ+fChQtjxozx9PSUl5ffvXs3PI51Cd98VFJSUjI0NCQIQl9fnyAIW1tbVBNASUmpoKCgqakpMTFRQ0MjLCwMuQ4kJSXfv3+Pfjt06FC0TX19/bq6OoIgjI2NCwsLDx48GBgYqKuru3XrVvJOU1JShIWF7ezs0MGMHj26uLgYP/IPGzYMSwDa2rRp0w4ePFhRURETE/Pff/+JiIhw80rAwcEhNjYWeRhCQkJGjx7N9oXc3Fy2tKNDhw5FCwYGBj9M6aesrIxspaen97XF0Hesra2Rm8vKykpMTCwxMfHTp09VVVXS0tLICSMhIVFWVobrAA8fPhxlOmR75LKzs0P5AvX19Q0MDMh+IQ7ExsbKyspWV1eHhoYiR3x8fDyLxXJwcDh16lRtbW1cXNy9e/eEhYW/9mtxAF1uISGhmJgYVVXVgoKC0NDQyMhIFRWVuLg4giBMTU39/f3Pnz8fExMzffr0adOm4d+qqKggLxbQQ0brd+7csba2njhxIkEQmzdvXrBgQUxMjI2NDT/2BZEwHCgoKEDeADK4IgrSX5yPEzmyKisrWSxWRkYGFixTU1Ps0NDS0vpftxMWRqNyAwODI0eOvHjx4urVq56eng4ODqtXryZ7ljU1NfGLvr59+yJfPConpKKigreGxq05OTnHjx8vKSkxMTExNTUVExPjJqCif//+6urqYWFhurq6NTU1WLIxpaWlRkZG5E/wu9PvdR7yQARbDJkIPwqQb2C6urr4XNTU1Orq6srLy4WEhGJjY8kqiYfnioqKbHtsampqaGhAdw5sLi5fdKM6lK9fv8afDBo0CBUiPnr0aG5urrGxMfk6fg82ayspKeHtMxgM8vZNTExYLNaMGTOUlZVDQ0OPHDlCpVKXLVuGy4BIS0vj1wNAT5D1z58/47I4IiIiBgYGGRkZfJJ1iIT5Hh8+fMjJyVm2bBk3Q3jyyJRKpdra2uIXjPHx8VizvgYNvVevXr1y5cqoqKjDhw///PPP+FahrKyclZVVU1MjKyuLXmZSKBRtbW30wvNrLly4oK2t7eHhQaVSWSzWvXv3uHzOc3BwCA8Pz8/Pt7e3/zoCR15evrCwkPMWhISEUMg/sk9xcTFWXm7eD2VnZ6OF2tra3NzcPn36KCsrM5nMOXPmoJE+jUZLSkrS1tbGA3Y2xMXFpaWl379/b2Zmhj5JTEzkMphETU2NTqdv3boVnXtZWVlhYaGkpKSHh4eUlNStW7dQdYHg4OCv7UmlUnHYe3Fx8fe2n5SUtGPHDmycyspKCoWCfFAjR45sbGy8fv26t7f36NGj0c2vsLAQ3w6BnuCEKS4uJl9RaWlpPk0DAcjQaLTMVhITEx8/fnz8+PHhw4d/PVrnjIiIyE8//fTo0aPMzEwWixUQEHDkyBEOQ+bc3Ny9e/cWFhZSqVSk5uR0/OjF3a1bt+rr6wsLCx8/fmxhYcGhKIGoqKiYmBiFQmEymdeuXWtubiYHXZCRlpZmsVh5eXlo/Pvzzz+npKSEh4d/00vet29fckjPN1FRUaFQKCEhISwWKyYm5sOHD22yG/J+NDU13blzR15efsiQIUZGRlpaWjdu3CgrK6PRaJcuXfLx8eGcAOvnn38OCQlJS0uj0+mPHz+uqqoaMGAAN3u3s7NjMBg3btxAI/QjR468fPkSlwBF98gHDx6gI0HWq62tLS8vZ7FYqqqq0dHR9a3cvn37m9sfOXJkVVXV3bt3m5ubq6qq9u/fHxERgSIs//rrr4aGBjExMSkpKRkZGRy8kJeXhx7OgB4yWm9sbCSPmNA4iE/78vX1nTt3Lvhh0EPShg0bkDgqKyuPHz8evaxrK0uXLj137tymTZskJSVVVVXXrl3LoQaNjY3Nzz///NtvvzGZzObmZjc3N/KAVEZGZteuXZ6eni9evCAIwtLScvPmzRx2PXv27PPnzy9cuJDBYIwa9f+xd/cxTZ1vH8BPCy1v5U0hIhYLCBYpKugP0KiRiTLUubnpfNkGihKjc62aQVQcGWZDgqgoHcrQ6KgOmJrMl0AWpxLRCWMoTJBB5U1EWqTQUqAvtj198uwkJw0o4vb4C/J8P3+V07v3Ob1Lr3Od+1znNGLu3Lnt7e0vyx99fX23b9+elJQ0Z86c8ePHT58+XS6XU0U7g4SGhp4/f95gMAwTVd3c3NauXfvdd98dP37cx8cnOjr6ta5dCgsLy87ONhgMTk5Oe/fupe5msXfv3mPHjlE/ue7t7S0SiYYv2YqJidHpdLt372az2c7Oznv27KHOgrySm5vb3r17jx8/XlxcbG1tHRQURJ3GXLVqVVZWVkxMDDUIkZGR1O7N39+fxWLFxcVlZ2d/9NFHaWlpn332mY2NTUxMTE1NzdD+vby8EhIScnNzL168aG1tHRwcTF0x8Mknn2RlZW3evJnaVSQkJNAZRkdHxxs6QP//YDTeE2bTpk2xsbERERHUnykpKXw+f+RXjrwW3BPmzeX+Op2OmjwZCaVS6eLi8rL5it7eXjs7uxFeoESdaRzJrxJqNBo7Oztqpfv37xcIBC/bkyUnJ8+fP//dd98dvkODwaDVakf+ri2RJNnb2zt0F6jRaBgMxivntWlGo3FgYOCfzWCoVCpHR8dBWc7LBl+j0djb2w/zwhd+yk5OToOaDd3g4uLiioqK0Vx7jUmY1+bu7t7R0WE5/Yry1bcOm81+rehmef3RUM7OziO/6NTFxWWEvzRrb2/PYDAaGhqKiooePny4ZMmSl7WMj4+/fPnyK3MgFov1z2I6fRHWCzdy5DGdOun6j2elXVxchobmlw0+HdNf9sIXfspDmw3aYJIkr169SqXwMHbCekREBH0FXVlZmUqlCgkJeUPrQiUMlJSUFBcXf/nll8NEQx6PFxISUllZieF60yoqKkJDQ9/S+zRgEma44/eUlBSpVOrj49PY2Lh9+3a67AkAAN6+sE4diNXW1qpUqlmzZnE4HHxOAABvd1j/r0ElDAAgrI8pqIQBgDEGN+YFAEBYH0NQCQMAYwwDPy4FAIBsHQAAENZHpbNnz/43f1MfAOBNQyUMKmEAANk6AAAgrI9OqIQBgDEGlTAAAMjWAQAAYX10QiUMAIwxqIRBJQwAIFsHAACE9dFpzZo1qIQBgLEElTAAAMjWAQAAYX10unDhAkmS+D8AgDEDlTCohAEAZOsAAICwPjqhEgYAxhhUwgAAIFsHAACE9dEJlTAAMMagEgaVMACAbB0AABDWRydUwgDAGINKGAAAZOsAAICwDgAACOsAAICwDgCAsD4MpVIpFou/+OKLjRs3ZmRk3LhxYyT9PnjwgHqQl5dXVFT0bzaxu7s7ISGhv7//TY9FamrqvXv33lz/3d3dSUlJ69evv3bt2qCxEolEn376aUtLy2t1SJJkbW0t/okB4DXCektLi1AolMlka9eu3bVrl4eHx+nTp8+cOTP8q8rKyrKysqjH48aNc3Z2/jebaDAYpFKpyWR602PR1NTU29v75vq/dOmSUqnct29feHi45fK8vLwJEyYkJydzudzX6vDKlSsFBQX4JwaAkYZ1s9mcnZ0dEBCQmpq6YMGCmTNnxsTE7Nu37+rVq1S+SZKk0WgkCKKnp0ev19MppE6nM5vNz58/Jwhi6dKlc+fOJQjC9DcqaTUYDFTjvr4+rVY7aL0qlUoul78yjlOdGAyGzs5OywSW7pwgCKPRSPXzwrX39/drNJqh3XZ2dg6t+1QoFGq1etCKzGZzd3f3Czevv7/fsh+j0ahUKgMDA/l8vuV+zmAwqFSq8PBwPz8/FotF9SyXy3U63aAOnz9//vTpU3qDTSaTTqej36/BYKBHjB586lXUu6a3xGQyyWQyugG9UC6XW75BAHhLWQ/zXENDg1QqFYvFDAaDXhgYGLhw4cKysrKoqKjbt29fuXLFwcGhublZp9NFRkZu27atvr4+OzvbYDDExsbm5OQcP37c09Nz48aNp0+f7u/vb2tra29vN5vNCQkJlZWVpaWlJEkuX748Li6OIIi6urrMzEy1Wm0ymZhMpkgkmj9//ssmNDZt2rRhw4aCggKz2Txu3Livvvpq8uTJd+7ckUgkp06dopqlp6dzudwNGzbk5+fLZLKurq7W1laz2SwUChsaGq5fv06S5OLFi7du3Uq1r6ioyMnJIQjCwcEhMTExMDCQyuIzMjIUCoXRaJwxY8auXbtcXV0rKyvz8vLc3d3v378fFxf34Ycf0tvW19d37NixiooKW1tbOzs7kUg0e/bs9PT0yspKBoNRXl7+448/0o03b96sUqlycnLKysqSk5Pv3r0rFoupXWN0dHR8fDyLxdJqtdnZ2Xfu3OFwOGq1es6cObt37759+/aFCxdMJtPmzZslEolIJFq1atXixYsJgmhtbd2xY8fFixeNRuO6detWrlx56dKl0NDQ5OTk69ev5+bmMhgMvV6/cuXK2NhYJpNZV1d38OBB/d98fX337Nnj5uaG7wbAGMzWZTKZtbW1l5fXoOV+fn5SqZR6/OjRo8mTJ585c+bw4cPl5eVFRUWBgYEikcjd3b2wsNDFxcXyhaWlpbGxsYWFhZGRkWlpaWw2++zZs19//fXVq1e1Wi1JkpmZmQsWLMj/28KFC0+fPj38wcSff/6Zm5ubn5/v7Ox88eLF4d/q3bt3V69e/dNPPy1btuzIkSPPnz/Py8tLS0v75Zdf6CxVKpWmpqZKJJKwsDA60h04cCA4OFgikeTm5jKZzBMnTlCNnzx5wuVyxWJxRESE5YpycnIUCoVYLJZIJFFRUWlpaXK5fN++fQsWLIiKirKM6QRBSCQSd3f3HTt2JCcnt7e3Z2RkbNy4MT8//9ChQw8ePLh8+TJBEMXFxfX19RKJ5Ny5cwcPHiwvL6+qqoqIiFi3bp1AIJBIJK+cXMrKyoqLi3v06JFYLBYKhQUFBQcOHCgtLf3111+pWaBFixbl5+cXFBSQJDnC0ycA8PaF9f7+fkdHRyZzcJvx48cPDAxQx/4MBmP16tUsFovH4/3nP/8pLy8fpkM/P7+QkBArK6vQ0FCz2fz+++/b2NgIBAKSJFtaWhgMxv79+9evX29lZaVUKl1cXFQq1fBbHxkZ6erqymazZ8+e3dXVNXzjyZMnh4eHM5nMsLAws9m8YsUKOzs7f39/FovV2NhItZk3b56/v7+tre0HH3zQ09NTW1tbVVWlVCqXL1+u1+tZLFZ0dHR5eTk9E7Ju3Toej+fq6mo57fPbb79FR0fzeDxbW1vq5gR3794dyYdx48YNDw+PsLAwtVrt5uY2f/78kpISgiCWLFmSnp7u5OSk0WiMRiOHw3nlyFhasWKFt7c3l8u9du3a1KlTg4KCent7J06cGB4efvPmTYIgnJycqqurf//9d7PZfOTIkbVr1+KLATA2J2GmTJmiVCr1er2NjY3lcrlc7u3tTU0E83g8OiXn8/nV1dXDdEgf2tvY2LBYLA8Pj//dsfyNJEkGg6FQKE6cONHY2GhnZ+fu7v7KW+by+Xzqga2t7Svn4sePH0+vnSAI+iiExWLRKxIIBNQDT09PR0fH7u7ugYEBkiQTExPpfuzt7aldiIODA4fDGbSWrq4ukiSnT59Od87n83t6ekbyYchkss7Ozm3bttFL2Gw2NUTnz5+vqqpSqVT+/v4mk2n4kRl0YmDChAl0/01NTZb9Uzuk7du3nzp1KjMz02g0BgcHx8fHT5w4Ed8NgDEY1v38/KytrW/evLl06VJ6IUmSJSUlAQEB1J+dnZ1Go9Ha2po62B8+HAxN/C0pFIqUlJSPP/6YmsMpLS2tr69/xbHGkA4ZDAZ1FpfS09NDl5cMv3Y6KFMPlEplX1+fm5sbi8Vis9k//PADdfNekiRNJhOLxers7Hxhh46OjtTs9qRJk6gI29raOmfOnJF8GM7Ozr6+vocOHaL+NBgM1CrEYnFXV1diYqKvr6+VldX69euHhnUmk0m/caVSafkUfS8zZ2fnGTNmpKSk0P1TTzk4OOzcudNsNldVVRUWFmZkZBw5cgTfDYAxOAnDZrOFQqFEIvnjjz+oJXq9Pjs7W6PRrFmzhlqi1WorKysJghgYGLh//35wcDCVOw8qtBgJ6pxkVFSUu7u7yWS6efOm2Wx+3bpGJycntVpNRee2trbHjx+/1svv3LlD1TiWlpZyOBw+nx8UFGQ0GulK88LCQpFIZLnnGITD4fj5+d26dYtqQ6fYI1n7rFmzpFIpvTNLTU2lwqtMJps1a5a/v7+VlVVZWRk9A2Y5zhwOp7m5mXp8+/btF/YfEhJSXV395MkTqvQlKSnp+++/JwgiMTHx559/ZrFYYWFh8+bNG1odBABjJFsnCOKdd97p6+vLzMx0cHBwdXVta2tzdXX95ptv6AkNe3v7EydOFBUVNTY28vl8qiCEy+X29fVt2bJl//79I9+UgIAAf3//pKSkgICApqYmKhQOSjxfSSAQcLncXbt2+fj4PHv2bNq0aa87Ilu3buVyuc3NzTt37nT425YtW3Jzc0tKSthsdmtra1JSEnV08jJCofDbb7+Nj4+fNGlSTU1NfHz8CMN6eHh4dHT07t27Z86cqVAoNBrN4cOHCYJ47733Tp061d7ertVqNRqNr68vNbfu7e198uTJzz///OjRo8uWLTt69GhrayvV4IX9L1q0qLq6WigUBgcHy2QygiBiYmIIgoiNjU1PT793756NjU1DQ4NQKMQXA+DtNaIb85Ik2dzcLJfLAwICLEvfbt26de7cucOHD9fU1Hh4eEyZMoV+qqOjo62tLSgoqLu728bGxsPDo6Ojw2QyUTPa/f39jx8/pieya2trfXx8HBwczGZzU1NTR0eHn5+fp6fnw4cPvby8bG1tGxoapk2bZhlMDQZDfX19QEAANcWvUCh6e3upDTAYDLW1tXq9PigoSK1WM5lMDw8PuVyu1+t5PB5BEBqNpqmpiZ7+rqur8/LycnR0rK+v53K57e3t1P7A3d3d8khCKpVyOJzAwEBqM9Rq9dOnT1+229DpdH/99ZdWq502bRp9QrW9vZ3JZHp6eg5qXF9fP3HiRLqYva2traWlhcvl+vr60qWlKpWqpqbG2dlZIBA8e/ZMp9P5+PhQ14s9e/YsJCSEzWbLZDKpVMrlcr29vevq6gQCgdlsrqurmzp1quXZkZaWlra2Nm9vb2o06PGsq6sbGBiYOnUqqhsBxn5YfxkqrJ88eRLjCAAwSvyrW33xeLwVK1ZgEAEAxki2DgAAYypbBwAAhHUAAEBYBwAAhHUAAIR1AABAWAcAAIR1AABAWAcAAIR1AACEdQAAQFgHAACEdQAAQFgHAACEdQAAhHUAAEBYBwAAhHUAAPi/8T8BAAD//1PcjCPQe4VJAAAAAElFTkSuQmCC)

So, with the increase in the no. of dimensions, it becomes very cumbersome to calculate distance between observations as well, and hence, all the Machine Learning algorithm that relies on calculating distance between observations find it very troublesome to work with high dimensional data, such as segmentation and clustering algorithms like KNN, K-Means etc.

**-------------------------------------------------------------------------------------------------------------------------------------------------------------------**

## Question 3 - Algorithmic question
You are given a list of integers, A, and another integer s. Write an algorithm that outputs all the pairs in A that equal x.


```python
# findPair function take List and integer s = 4
def findPair(A,s):
    
    # length or size is store in length var of List A
    length = len(A)
    # initialize counter variable to 0
    counter = 0
    # declare pairList List to store all the pairs which has sum equal to 4
    pairList = []
    # outer for loop goes till length of List A
    for i in range(length):
        # Inner for loop goes till length of List A for every iteration of outer for loop 
        for j in range(i+1,length):
            # sum both elements which are adjacent to each other and store in pairSum
            pairSum = A[i]+A[j]
            # compare the pairSum in the list to s which is equal to 4
            if pairSum == s:
                firstPair = A[i]
                secondPair = A[j]
                # store the element of pair which has sum equal to 4
                strPair = "("+str(firstPair)+","+str(secondPair)+")"
                # add to the pairList List which contains only pair sum
                pairList.append(strPair)
    
    # return pairList to the main function             
    return pairList        

# Main function 
if __name__ == "__main__":
    
    
    returnPairList = []
    
    # declare and initialize the list A of integers numbers
    A = [7,-2,8,2,6,4,-7,2,1,3,-3]
    # declare and initialize the s equal to 4
    s=4
    # call to function findPair and pass the List and S
    returnPairList = findPair(A,s)
    print (str(returnPairList)[1:-1]) 
    
    # for i in range(len(returnPairList)):
    #     print(*returnPairList[i])
```

    '(7,-3)', '(-2,6)', '(2,2)', '(1,3)'
    

Group #16
* **Question 1**: Seyed Behdad Ahmadi
* **Question 2**: Mascolo Davide
* **Question 3**: Mossaab Moustaghit
