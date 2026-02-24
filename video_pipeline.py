
from google import genai
from google.genai import types
from PIL import Image, ImageFilter
import json
import re
import time
import random
import requests
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
from pydub import AudioSegment
import whisper
import os
import threading
from dotenv import load_dotenv
import pysrt

# Load env variables
load_dotenv("d:/JAK/.env")

# =====================================
# CONFIG CLASS
# =====================================

class VideoConfig:
    def __init__(self):
        self.GENAI_API_KEY = os.getenv("GENAI_API_KEY")
        
        # Split keys by comma for rotation
        deapi_env = os.getenv("DEAPI_KEY", "")
        self.DEAPI_KEYS = [k.strip() for k in deapi_env.split(",") if k.strip()]
        
        self.ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
        self.VOICE_ID = os.getenv("VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        self.SCENE_IMAGES = {
            "scene1": r"d:/gemini/front.png",
            "scene2": r"d:/gemini/left.png",
            "scene3": r"d:/gemini/right.png",
            "scene4": r"d:/gemini/back.png",
        }
        
    def update_images(self, front, left, right, back):
        # Allow overriding defaults with uploaded paths
        if front: self.SCENE_IMAGES["scene1"] = front
        if left: self.SCENE_IMAGES["scene2"] = left
        if right: self.SCENE_IMAGES["scene3"] = right
        if back: self.SCENE_IMAGES["scene4"] = back
        
        self.SCENE_FILES = {
            "scene1": "scene1.mp4",
            "scene2": "scene2.mp4",
            "scene3": "scene3.mp4",
            "scene4": "scene4.mp4",
        }
        
        self.FINAL_VIDEO = "final_reel_ad_9x16.mp4"
        self.FINAL_VIDEO_WITH_VOICE = "final_video_with_voice.mp4"
        self.OUTPUT_AUDIO = "final_voice.mp3"
        self.SAFE_AUDIO = "final_voice_safe.mp3"
        self.SRT_OUTPUT = "ainsta_caption.srt"
        
        self.TARGET_W = 432
        self.TARGET_H = 768
        self.MAX_WORDS = 3
        self.WHISPER_MODEL_SIZE = "small"

# =====================================
# VIDEO PIPELINE ENGINE
# =====================================

class VideoPipeline:
    def __init__(self, config: VideoConfig = None):
        if config is None:
            config = VideoConfig()
        self.cfg = config
        
        # Configure ImageMagick (Required for TextClip in Windows)
        change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})
        
        # New SDK Client
        self.client = genai.Client(api_key=self.cfg.GENAI_API_KEY)
        self.model_name = "gemini-2.5-flash" # Switched to 1.5-flash for better free tier quota
        
        # State management
        self.logs = []
        self.status = "Idle"
        self.progress = 0
        self.current_step = ""
        self.error = None
        self.generated_scenes = {}
        self.current_key_idx = 0

    def get_current_api_key(self):
        if not self.cfg.DEAPI_KEYS:
            return None
        return self.cfg.DEAPI_KEYS[self.current_key_idx % len(self.cfg.DEAPI_KEYS)]

    def rotate_key(self):
        self.current_key_idx += 1
        new_key = self.get_current_api_key()
        self.log(f"🔄 Rotating API Key... (Using key #{self.current_key_idx % len(self.cfg.DEAPI_KEYS) + 1})")
        return new_key

    def _auto_font_and_size(self, video_width):
        available_fonts = ["Arial-Bold", "Verdana-Bold", "Times-New-Roman", "Courier-New-Bold"]
        font_name = available_fonts[video_width % len(available_fonts)]
        font_size = int(video_width * 0.08) # Increased from 0.045 to 0.08 for bigger fonts
        return font_name, font_size

    def _resolve_position(self, position):
        if isinstance(position, tuple) and len(position) == 3 and position[0] == "axis":
            _, x, y = position
            return (int(x), int(y))
        return position

    def burn_captions(self, video_path, srt_path, output_path):
        try:
            self.log(f"Burning captions into {output_path}...")
            video = VideoFileClip(video_path)
            subs = pysrt.open(srt_path)
            
            font_name, font_size = self._auto_font_and_size(video.w)
            # Override for visibility if needed, or stick to auto
            font_color = "yellow" 
            stroke_color = "black"
            stroke_width = 2
            
            caption_clips = []
            for sub in subs:
                start_time = sub.start.ordinal / 1000
                end_time = sub.end.ordinal / 1000
                duration = end_time - start_time
                if duration <= 0: continue

                txt_clip = (TextClip(
                    sub.text,
                    fontsize=font_size,
                    font=font_name,
                    color=font_color,
                    stroke_color=stroke_color,
                    stroke_width=stroke_width,
                    method="caption",
                    size=(int(video.w * 0.9), None)
                )
                .set_start(start_time)
                .set_duration(duration)
                .set_position(("center", "bottom"))) # Default to bottom center
                
                caption_clips.append(txt_clip)

            final = CompositeVideoClip([video, *caption_clips])
            final.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=video.fps or 30)
            self.log("Captions burned successfully.")
            
            # Cleanup
            video.close()
            final.close()
            
        except Exception as e:
            self.log(f"Caption Burn Error: {e}")
            # If burning fails, fallback by just copying source to dest so pipeline completes
            try:
                import shutil
                shutil.copy(video_path, output_path)
                self.log("Fallback: Copied video without captions due to error.")
            except:
                pass

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        print(entry)
        self.logs.append(entry)

    def clean_json(self, text: str):
        text = re.sub(r"```json|```", "", text).strip()
        try:
            return json.loads(text)
        except Exception as e:
            self.log(f"JSON Parse Error: {e}")
            return {}

    def convert_to_vertical_safe(self, image_path, output_path):
        try:
            self.log(f"Processing image: {image_path}")
            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            bg = img.resize((self.cfg.TARGET_W, self.cfg.TARGET_H), Image.LANCZOS)
            bg = bg.filter(ImageFilter.GaussianBlur(30))

            scale = min(self.cfg.TARGET_W / w, self.cfg.TARGET_H / h)
            fg = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            x = (self.cfg.TARGET_W - fg.width) // 2
            y = (self.cfg.TARGET_H - fg.height) // 2

            bg.paste(fg, (x, y))
            bg.save(output_path)
            return output_path
        except Exception as e:
            self.log(f"Error converting image: {e}")
            raise

    # STEP 1
    def step_generate_prompts(self):
        self.status = "Generating Prompts"
        self.progress = 10
        self.log("Asking Gemini to design scenes...")
        
        images = []
        for k, v in self.cfg.SCENE_IMAGES.items():
            try:
                images.append(Image.open(v))
            except FileNotFoundError:
                self.log(f"Error: Image not found {v}")
                raise

        prompt = """
        You are an elite cinematic advertisement director and AI video engineer.
        You are given 4 images of the SAME product from different angles.
        Infer product category, material, surface behavior, scale.

        Rules:
        - Same dark premium studio
        - Soft volumetric fog
        - Controlled rim lighting
        - Glossy floor reflections
        - Vertical 9:16 framing
        - No distortion

        Scene logic:
        1. Hero reveal
        2. Side geometry
        3. 3D orbit / depth
        4. Important detail close-up

        Return STRICT JSON ONLY:
        {
          "scene1": "",
          "scene2": "",
          "scene3": "",
          "scene4": ""
        }
        """
        
        try:
            # New SDK call
            # contents can be a list of strings and PIL Images directly
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt] + images
            )
            self.generated_scenes = self.clean_json(resp.text)
            self.log("Scenes designed successfully.")
            return self.generated_scenes
        except Exception as e:
            self.log(f"Gemini Error: {e}")
            raise

    # STEP 2
    def step_generate_video_scene(self, scene_key, prompt):
        self.status = f"Generating {scene_key}"
        out_file = self.cfg.SCENE_FILES[scene_key]
        image_path = f"safe_{scene_key}.png"
        
        # Prepare Image
        self.convert_to_vertical_safe(self.cfg.SCENE_IMAGES[scene_key], image_path)
        
        self.log(f"Generating video for {scene_key}...")
        
        url = "https://api.deapi.ai/api/v1/client/img2video"
        
        try:
            files = {"first_frame_image": open(image_path, "rb")}
        except FileNotFoundError:
            self.log(f"Failed to open image {image_path}")
            return

        data = {
            "prompt": prompt,
            "width": self.cfg.TARGET_W,
            "height": self.cfg.TARGET_H,
            "fps": 30,
            "frames": 120,
            "steps": 1,
            "guidance": 8,
            "seed": random.randint(1, 99999999),
            "model": "Ltxv_13B_0_9_8_Distilled_FP8",
            "motion": "cinematic",
        }

        # RETRY LOOP FOR ROTATION
        max_retries = 5
        for attempt in range(max_retries):
            current_key = self.get_current_api_key()
            if not current_key:
                self.log("Error: No DEAPI keys found in .env")
                return

            headers = {"Authorization": f"Bearer {current_key}"}
            
            # Re-open file pointer for each attempt to avoid read errors if seeker moved
            files["first_frame_image"].seek(0)
            
            try:
                r = requests.post(url, data=data, files=files, headers=headers)
                j = r.json()
                
                # Check for specific error message
                if "message" in j and "Too Many Attempts" in j["message"]:
                    self.log(f"⚠️ Rate Limit hit on Key #{self.current_key_idx % len(self.cfg.DEAPI_KEYS) + 1}")
                    self.log("⏳ Waiting 20s before switching key...")
                    time.sleep(20)
                    self.rotate_key()
                    continue # Retry with new key
                
                if "data" not in j:
                    self.log(f"API Error: {j}")
                    # If it's another error, maybe we shouldn't retry infinitely, but let's try rotating once just in case?
                    # For now, strict on "Too Many Attempts", break on others to avoid burn
                    return

                request_id = j["data"]["request_id"]
                status_url = f"https://api.deapi.ai/api/v1/client/request-status/{request_id}"
                
                while True:
                    res = requests.get(status_url, headers=headers).json()
                    prog = res["data"].get("progress", 0)
                    self.progress = int(prog) # Update global progress for UI
                    
                    if prog >= 100:
                        video_url = res["data"]["result_url"]
                        with open(out_file, "wb") as f:
                            f.write(requests.get(video_url).content)
                        self.log(f"Saved: {out_file}")
                        return # Success!
                    
                    if res["data"].get("status") == "failed":
                         self.log(f"Generation Failed: {res}")
                         return
                    
                    time.sleep(2)
                
                break # Break retry loop if successful (though return handles it above)

            except Exception as e:
                self.log(f"Video Gen Error: {e}")
                time.sleep(2)


    # STEP 3
    def step_merge_scenes(self):
        self.status = "Merging Scenes"
        self.progress = 0
        try:
            self.log("Merging video clips...")
            clips = [VideoFileClip(self.cfg.SCENE_FILES[k]) for k in self.cfg.SCENE_FILES]
            final = concatenate_videoclips(clips, method="compose")
            final = final.resize((self.cfg.TARGET_W, self.cfg.TARGET_H))
            final.write_videofile(self.cfg.FINAL_VIDEO, fps=30)
            self.log(f"Final video ready: {self.cfg.FINAL_VIDEO}")
        except Exception as e:
            self.log(f"Merge Error: {e}")

    # STEP 4: Voiceover & Subtitles
    def step_finalize_video(self):
        self.status = "Finalizing (Voice & Subs)"
        self.progress = 0
        
        try:
            self.log("Analyzing output video for script...")
            clip = VideoFileClip(self.cfg.FINAL_VIDEO)
            duration = round(clip.duration, 2)
            clip.close()

            # Script Gen
            with open(self.cfg.FINAL_VIDEO, "rb") as f:
                video_bytes = f.read()

            prompt = f"""
            You are a professional cinematic advertisement voiceover writer.
            STRICT RULES:
            - Script MUST fit within {duration} seconds
            - Use <emphasis> and <break> tags
            - Natural sentences only
            - Return only formatted text
            """
            
            # New SDK call for video bytes
            r = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
                ]
            )
            script_text = r.text.strip()
            self.log(f"Generated Script: {script_text}")

            # Voice Gen
            self.log("Generating Voiceover...")
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.cfg.VOICE_ID}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.cfg.ELEVEN_API_KEY
            }
            data = {
                "text": script_text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.6, "similarity_boost": 0.7}
            }
            audio = requests.post(url, json=data, headers=headers).content
            with open(self.cfg.OUTPUT_AUDIO, "wb") as f:
                f.write(audio)

            # Safety Audio padding
            audio_seg = AudioSegment.from_mp3(self.cfg.OUTPUT_AUDIO)
            if len(audio_seg) / 1000 < duration:
                audio_seg += AudioSegment.silent(duration=int((duration * 1000) - len(audio_seg)))
            audio_seg.export(self.cfg.SAFE_AUDIO, format="mp3")

            # Attach Audio
            self.log("Attaching Audio...")
            video = VideoFileClip(self.cfg.FINAL_VIDEO)
            audio_clip = AudioFileClip(self.cfg.SAFE_AUDIO)
            final = video.set_audio(audio_clip)
            final.write_videofile(self.cfg.FINAL_VIDEO_WITH_VOICE, codec="libx264", audio_codec="aac")
            
            # Cleanup
            video.close()
            audio_clip.close()
            final.close()

            # Captions
            self.log("Generating Properties...")
            model = whisper.load_model(self.cfg.WHISPER_MODEL_SIZE)
            result = model.transcribe(
                self.cfg.FINAL_VIDEO_WITH_VOICE,
                word_timestamps=True,
                verbose=False
            )
            # ... (Existing SRT logic implementation would go here, simplified for brevity but assuming same logic)
            # Re-implementing the SRT logic briefly:
            self._generate_srt(result, self.cfg.SRT_OUTPUT)
            
            # Burn Captions
            self.log("Burning Captions...")
            final_captioned = "final_reel_captioned.mp4"
            self.burn_captions(self.cfg.FINAL_VIDEO_WITH_VOICE, self.cfg.SRT_OUTPUT, final_captioned)
            
            self.log(f"Final Video Complete: {final_captioned}")
            self.status = "Completed"
            self.progress = 100

        except Exception as e:
            self.log(f"Finalize Error: {e}")
            self.status = "Error"

    def _generate_srt(self, result, output_path):
        # Helper for SRT generation
        def format_time(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        index = 1
        srt_lines = []
        max_words = self.cfg.MAX_WORDS

        for segment in result["segments"]:
            words = segment.get("words", [])
            i = 0
            while i < len(words):
                chunk = words[i:i + max_words]
                if not chunk: break
                start = chunk[0]["start"]
                end = chunk[-1]["end"]
                text = " ".join(w["word"].strip() for w in chunk)
                srt_lines.append(f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n")
                index += 1
                i += max_words
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))
        self.log(f"SRT saved to {output_path}")

    # MAIN RUNNER
    def run_full_pipeline(self):
        try:
            scenes = self.step_generate_prompts()
            for key in scenes:
                self.step_generate_video_scene(key, scenes[key])
                time.sleep(5) # Brief pause
            
            self.step_merge_scenes()
            self.step_finalize_video()
        except Exception as e:
            self.log(f"Pipeline Failed: {e}")
            self.status = "Failed"

