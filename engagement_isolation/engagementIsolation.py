#!/usr/bin/env python3
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import cv2
import sys, os, time
sys.path.insert(0, '..' + os.path.sep + 'yolov3_object_tracking')

from yolov3_testing import load_yolo, detect_objects, get_box_dimensions


class Isolator:
	def __init__(self, file_path, out_dir, clip_length=120):
		self.file_path = file_path
		self.file_name = file_path.split(os.path.sep)[-1]
		self.clip_length = clip_length
		self.out_dir = out_dir
		self.video = VideoFileClip(file_path)


	def refresh_video(self):
		self.video = VideoFileClip(self.file_path)

	def _cleanup_file(self, path):
			print("cleaning up file: " + path)
			os.remove(path)

	def get_frames(self):
		video = cv2.VideoCapture(self.file_path)
		while video.isOpened():
			rete, frame = video.read()
			if rete:
				yield frame
			else:
				break
		video.release()
		yield None

	def center_crop(self, img, width, height):
		crop_width = width if width < img.shape[1] else img.shape[1]
		crop_height = height if height < img.shape[0] else img.shape[0]

		mid_x, mid_y = int(img.shape[1]/2), int(img.shape[0]/2)
		cw2, ch2 = int(crop_width/2), int(crop_height/2)
		return img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

	def new_clip(self, clip_counter, size):
		return cv2.VideoWriter(self.clip_name(clip_counter), 
			cv2.VideoWriter_fourcc(*'mp4v'), 15 , size)

	def clip_name(self, clip_counter):
		return self.out_dir + os.path.sep + self.file_name.split('.')[0] + '_' + str(clip_counter) + '_engagement.mp4'

	def parse_video(self, crop_width, crop_height):
		clip_counter = 0
		recording = 0
		wrote_on_prev_frame = False
		video_out = None
		clips = {}
		model, classes, colors, output_layers = load_yolo(
			'..'+os.path.sep+'yolov3_object_tracking' + os.path.sep + 'yolov3.weights',
			'..'+os.path.sep+'yolov3_object_tracking' + os.path.sep + 'yolov3.cfg',
			'..'+os.path.sep+'yolov3_object_tracking' + os.path.sep + 'coco.names')


		counter = 0
		for frame in self.get_frames():
			if frame is None:
				break
			counter += 1
			frame = self.center_crop(frame, crop_width, crop_height)
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			height, width, channels = frame.shape

			_, outputs = detect_objects(frame, model, output_layers)
			detections, confs, class_ids = get_box_dimensions(outputs, height, width)



			for i in range(0, len(detections)):
				if confs[i] > 0.5:
					recording = self.clip_length
					if not wrote_on_prev_frame:
						clip_counter += 1
						clips[self.clip_name(clip_counter)] = {
							"start": counter
						}
						
			if recording > 0:
				recording -= 1
				wrote_on_prev_frame = True
			elif wrote_on_prev_frame:
				clips[self.clip_name(clip_counter)]["end"] = counter
				wrote_on_prev_frame = False

		curr_clip_name = self.clip_name(clip_counter)
		if clips[curr_clip_name] and clips[curr_clip_name]["end"] is None:
			clips[curr_clip_name]["end"] = counter
		return clips


	def cut_clips(self, clips):
		for file_name in clips:
			clip = clips[file_name]
			# ffmpeg_extract_subclip(self.file_path, int(clip["start"] / self.video.fps), int(clip["end"]/ self.video.fps), targetname=file_name)
			v_clip = self.video.subclip(int(clip["start"] / self.video.fps), int(clip["end"]/ self.video.fps))
			v_clip.write_videofile(file_name, fps=self.video.fps)
			# self.refresh_video()







def main():
	print("Running main")
	iso = Isolator(sys.argv[1], sys.argv[2], 120)
	iso.cut_clips(iso.parse_video(200, 200))


if __name__ == "__main__":
	main()
