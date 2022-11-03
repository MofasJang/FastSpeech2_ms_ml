import pyaudio  
import wave  
def playwav(file):
    #define stream chunk   
    chunk = 1024  
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    
    #open a wav format music  
    f = wave.open(file,"rb")  
    
    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    #read data  
    data = f.readframes(chunk)  

    #play stream  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  

    #stop stream  
    stream.stop_stream()  
    stream.close() 
    f.close()

    #close PyAudio  
    p.terminate()  