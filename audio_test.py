# Standard library
import logging

# Third party
import zignal

# Internal
import zoundcard


def play():
	# Another way of using a soundcard is by first creating an instance and
	# manually calling the open() function. The close() function *must* be
	# called in a controlled fashion. This usually means that the usage is
	# wrapped in a try-except-finally clause.

	fs = 44100
	dur = 10
	x  = zignal.Sinetone(f0=424, fs=fs, duration=dur)
	x2 = zignal.Sinetone(f0=282, fs=fs, duration=1,phasedeg=0)
	for i in range(0,10):
		x3 = zignal.Sinetone(f0=282, fs=fs, duration=1,phasedeg=i*18)
		x2.concat(x3)
	x.append(x2)


	x = x.to_mono()
	x.convert_to_integer(targetbits=16)

	snd = zoundcard.PA( device_out='Headphones (BT5.0-Audio Stereo)')
	#x2.plot(plotrange=(0,10))
	#print(snd)

	snd.open()
	try:
		snd.play(x)
	finally:
		snd.close()


if __name__ == '__main__':
	logging.basicConfig(
		format="%(levelname)-8s: %(module)s.%(funcName)-15s %(message)s",
		level="DEBUG",
		)
	# some libraries are noisy in DEBUG
	logging.getLogger("matplotlib").setLevel(logging.INFO)
	logging.getLogger("PIL").setLevel(logging.WARNING)

	play()

	print('++ End of script ++')