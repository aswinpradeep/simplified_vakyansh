import wave
import os
import subprocess
from pydub import AudioSegment
import collections
import webrtcvad
import tqdm
import numpy as np
import datetime
import contextlib
import torch
import torch.nn.functional as F
from fairseq.data import Dictionary
from inference_lib.w2l_viterbi_decoder import W2lViterbiDecoder
# from inverse_text_normalization.run_predict import inverse_normalize_text
from inference_lib.w2l_kenlm_decoder import W2lKenLMDecoder
# from fairseq import utils
# from inference_lib.utilities import get_args, get_results, load_cpu_model ,load_gpu_model

 

def read_audio(in_file,type):
    import datetime
    with wave.open(in_file, 'rb') as f:
        return f.readframes(f.getnframes())


def create_wav_file_using_bytes(file_name, audio):
    with wave.open(file_name, 'wb') as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000.0)
        file.writeframes(audio)
    return os.path.join(os.getcwd(), file_name)


def media_conversion(file_name, duration_limit=5):
    dir_name = os.path.join('/tmp', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(dir_name)

    subprocess.call(["ffmpeg -i {} -ar {} -ac {} -bits_per_raw_sample {} -vn {}".format(file_name, 16000, 1, 16, dir_name + '/input_audio.wav')], shell=True)

    audio_file = AudioSegment.from_wav(dir_name + '/input_audio.wav')

    audio_duration_min = audio_file.duration_seconds / 60

    if audio_duration_min > 5:
        clipped_audio = audio_file[:300000]
        clipped_audio.export(dir_name + '/clipped_audio.wav', format='wav')
    else:
        audio_file.export(dir_name + '/clipped_audio.wav', format='wav')

    os.remove(dir_name + '/input_audio.wav')
    print(dir_name)
    return dir_name

def noise_suppression(dir_name):
    
    """ function to execute fb denoiser. 
    it accepts arguments --dns48(prebuiltmodel) , input & file directories, no of threads.
    function takes in all the wav files in the input directory,
    ( necessarily with sample rate of 16000) and saves the enhanced files to the output directory
    in the format filename+_enhanced.wav'
    """

    subprocess.call(["python -m denoiser.enhance --dns48 --noisy_dir {} --out_dir {} --sample_rate {} --num_workers {} --device cpu".format(dir_name, dir_name, 16000, 1)], shell=True)


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames, start_time, end_time):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        #sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                start_time.append(ring_buffer[0][0].timestamp)
                #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                end_time.append(frame.timestamp + frame.duration)
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        end_time.append(frame.timestamp + frame.duration)
        #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    #sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def extract_time_stamps(wav_file):
    start_time = []
    end_time = []
    audio, sample_rate = read_wave(wav_file)
    vad = webrtcvad.Vad(3)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames, start_time, end_time)
    chunks = 0
    for i, segment in enumerate(segments):
        chunks = chunks + 1
    if chunks != len(start_time):
        print("Error: Segments not broken properly")
        exit
    return start_time, end_time

def formatSrtTime(secTime):
	sec, micro = str(secTime).split('.')
	m, s = divmod(int(sec), 60)
	h, m = divmod(m, 60)
	return "{:02}:{:02}:{:02},{}".format(h, m, s, micro[:2])

def get_feature(wav, sample_rate):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    # wav, sample_rate = sf.read(filepath)
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats, sample_rate)
    return feats


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence





def get_results_from_chunks(wav_data, dict_path, generator, use_cuda=False, w2v_path=None, model=None):
    sample = dict()
    net_input = dict()
    feature = wav_data
    target_dict = Dictionary.load(dict_path)
 
    # model[0].eval()
    model.eval()

    if generator is None:
        generator = W2lViterbiDecoder(target_dict)
           
    net_input["source"] = feature.unsqueeze(0)

    padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)

    net_input["padding_mask"] = padding_mask
    sample["net_input"] = net_input
    # sample = utils.move_to_cuda(sample) if use_cuda else sample

    with torch.no_grad():
        hypo = generator.generate(model, sample, prefix_tokens=None)
    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())
    text=post_process(hyp_pieces, 'letter')

    return text

def response_alignment(response, num_words_per_line=25):
    
    aligned_response = []
    
    if len(response.split(' ')) < 25:
        aligned_response.append(response)
    else:
        
        num_lines = len(response.split(' ')) //  25
        for line in range(num_lines):
            aligned_response.append(' '.join(response.split(' ')[25*line: 25*(line+1)]))
        aligned_response.append(' '.join(response.split(' ')[(line+1)*25:]))
    return aligned_response


def generate_srt(wav_path, language, model, generator, cuda, dict_path):
    start_time, end_time = extract_time_stamps(wav_path)
    original_file_path = wav_path.replace('clipped_audio_enhanced', 'clipped_audio')
    original_chunk = AudioSegment.from_wav(original_file_path)
    result = ''
    for i in tqdm(range(len(start_time))):
        if end_time[i] - start_time[i] > 80:
            result+=str(i+1)
            result+='\n'
            result+=str(formatSrtTime(start_time[i]))
            result+=' --> '
            result+=str( formatSrtTime(end_time[i]))
            result+='\n'
            result+='speech is not clear in this segment'
            result+='\n\n'
            continue
        chunk = original_chunk[start_time[i]*1000: end_time[i]*1000]
        float_wav = np.array(chunk.get_array_of_samples()).astype('float64')
        features = get_feature(float_wav, 16000)
        result+=(str(i+1))
        result+='\n'
        result+=str(formatSrtTime(start_time[i]))
        result+=' --> '
        result+=str( formatSrtTime(end_time[i]))
        result+='\n'
        response = get_results_from_chunks(wav_data=features, dict_path=dict_path, generator=generator, use_cuda=cuda, model=model)
        if language=='en-IN':
            response = response.lower()
        aligned_response = response_alignment(response, num_words_per_line=25)
        result+='\n'.join(aligned_response)
        result+='\n\n'
    return result

def apply_punctuation(self, text_to_punctuate, language, punctuate):
    result = text_to_punctuate
    if punctuate:
        punc_model_obj = self.punc_models_dict.get(language, None)
        if punc_model_obj != None:
            result = punc_model_obj.punctuate_text([text_to_punctuate])[0]
    return result

def apply_itn(self, text_to_itn, language, itn):
    result = text_to_itn
    if itn:
        enabled_itn = self.enabled_itn_lang_dict.get(language, None)
        if enabled_itn != None:
            result = inverse_normalize_text([text_to_itn], language)[0]
    return result

def get_srt3(file_name, model, generator, dict_path, audio_threshold=5, language='hi'):
    dir_name = media_conversion(file_name, duration_limit=audio_threshold)
    noise_suppression(dir_name)
    audio_file = dir_name + '/clipped_audio_enhanced.wav'

    result = generate_srt(wav_path=audio_file, language=language, model=model, generator=generator, cuda=torch.cuda.is_available(), dict_path=dict_path)
    return result


def get_srt2(file_name, language, model_path, dict_path):
    generator = None
    model = torch.load(model_path, map_location=torch.device("cpu"))


    if language == 'hi' or language == 'en-IN' or language == 'kn-lm':
        # generator = self.generators[language]
        generator = W2lKenLMDecoder(Dictionary.load(dict_path))

    # result = get_srt3(file_name=file_name, model=self.models[language], generator=generator, dict_path=self.dict_paths[language], language=language, denoiser_path=denoiser_path)
    result = get_srt3(file_name=file_name, model=model, generator=generator, dict_path=dict_path, language=language)

    res = {}
    res['status'] = "OK"
    res['srt'] = result
    return res

def get_srt1(file_name, language, model_path, dict_path, punctuate, itn):
    result = get_srt2(file_name, language, model_path, dict_path)
    print("Before Punctuation**** ", result['srt'])
    result['srt'] = apply_punctuation(result['srt'], language, punctuate)
    result['srt'] = apply_itn(result['srt'], language, itn)
    print("After Punctuation**** ", result['srt'])
    return result






#HARDCODINGS

PUNCTUATE = True
ITN = False
LANGUAGE = 'hi'
AUDIOFORMAT = 'MP3'

FILE_PATH = "/home/aswin/speech-text-dataset-generation/hindi/1920_202161/audio_chunks/chunk23.wav"
MODEL_PATH = "/home/aswin/Downloads/hindi/final_model_ajitesh_test.pt"
DICT_PATH = "/home/aswin/Downloads/hindi/dict.ltr.txt"

file_name = 'audio_input_{}.{}'.format(str("get_current_time_in_millis"), "audio_format.lower")
audiocontent = read_audio(FILE_PATH,"wav")

audio_path = create_wav_file_using_bytes(file_name, audiocontent)
response = get_srt1(audio_path, LANGUAGE, MODEL_PATH, DICT_PATH, PUNCTUATE, ITN)
srt=response['srt']
print(srt)
os.remove(audio_path)