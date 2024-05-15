import numpy as np
import itertools as it
import glob
from obspy.core import read, UTCDateTime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from obspy.io.sac.util import get_sac_reftime
from obspy.signal.cross_correlation import xcorr_pick_correction, correlate

import warnings
warnings.filterwarnings("ignore")
#plt.style.use('seaborn-paper')


def decorr_index(station_name, data_list, trace_len, bandpass_filter, windowing):

	'''
	Assuming is the same station we're going to measure the delay in time between 
	two similar events events using a moving window.  

	Template is an obspy stream with the waveform template
	stream_waves is an obspy stream with at least one waveform
	'''

	def time_Ppicks(trace):
		'''This function reads the Pwave arrival time picked by the analyst
		and returns that time in UTCDate format'''
		ref_time_event = get_sac_reftime(trace.stats.sac)
		ptime_pick = ref_time_event + trace.stats.sac.a

		return ptime_pick

	
	def prep_data(waveform):
		st = read(waveform)
		tr = st[0]

		# Creating a time_vector in seconds
		sampling_rate = tr.stats.sampling_rate
		delta = 1.0/sampling_rate

		# getting the p-wave pick time
		ptime_pick = time_Ppicks(tr)

		"""Removing the mean and trend for pre-processing the traces"""
		tr.detrend(type="demean")
		tr.detrend('linear')

		return tr, ptime_pick
	
	def back_in_time(Trace_a, Trace_b):
		''' Setting the starttime of both traces to 1970-01-01:00:00:00
		this step is necessary then for slicing and sliding'''

		back_time = UTCDateTime("1970-01-01:00:00:00")
		
		Trace_a.stats.starttime = back_time
		Trace_b.stats.starttime = back_time

		return Trace_a, Trace_b
		

	def rolling_window(a, b, wlen, stp):
		'''Function for computing the rolling window between 2 time series'''

		'''send them back in time'''
		pa, pb  = back_in_time(a, b)


		correlation_values=[]
		eq_dist = []
		for kwin, window in enumerate(pa.slide(window_length=wlen, step=stp, 
									include_partial_windows=False, nearest_sample=True)):
			tmpa = window.copy()

			'''Defining the start and end times'''
			stime = tmpa.stats.starttime
			etime = tmpa.stats.endtime
			
			'''Slicing trace B based on the number of time windows from the trace A '''
			tmpb = pb.slice(starttime=stime, endtime=etime, nearest_sample=True)

			'''Extracting the data from the Obspy Trace'''
			a_i = tmpa.data
			b_i = tmpb.data
			time_vector = np.arange(0, len(tmpa)) * 0.01
			'''Applying the moving window waveform correlation'''
			#cc = correlate_template(a_i, b_i, normalize='full', method='direct', mode='same', )
			cc = correlate(a_i, b_i, 50, method='direct')
			# Using EC 49 from Snider and Vrijlandt, 2005: Constraining Source Separation with Coda Wave interferometry
			# Calculate the coda wave interferometry Green's functions
			g1 = np.real(np.fft.ifft(np.abs(np.fft.fft(cc)) ** 2))
			g2 = np.real(np.fft.ifft(np.abs(np.fft.fft(b_i.data)) ** 2))
			# Calculate the distance between the two earthquakes in meters
			dist = np.sqrt(g1.max() / g2.max()) * 10000000
			cc_max_index = np.argmax(cc) - len(a_i)
			#time_shift = cc_max_index / 0.01
			#s2_t = np.var(time_shift)
			#vs = 3.0 # km/s
			#dist = np.sqrt(3 * s2_t / vs**2)
			eq_dist.append(dist)

			index_max_cc = np.argmax(cc)
			max_cc = cc[index_max_cc]
			correlation_values.append(max_cc)


		return correlation_values, eq_dist


	def processing(data_list, t_before_p, t_after_p, bandpass_filter, windowing, station_name):

		''' Generating a unique combination of waveforms to be used for cross-correlation'''
		unique_combinations = list(it.combinations(np.unique(data_list), 2))
		
		''' For each waveform combination we are going to pre-process the data:
			1. Removing the mean and trend
			2. Aligning the waveforms on the Pwave arrival. 
		'''

		Xcorr=[]
		Xvals=[]
		Xstds=[]
		for filter in bandpass_filter:
			print(f"*** Processing data in the frequency band: {filter[0]} - {filter[1]} Hz ***")
			
			'''This is a list containing all the moving window 
			correlations for all the possible unique waveform pairs'''
			xcorr_max=[]
			xvals_all=[]
			xtdiff=[]

			Relative_distance=[]
			Relative_distance_std=[]
			Relative_time=[]
			'''Looping over the unique pair of events'''
			file_out = f"Correlation_distance_results_{station_name}_{filter[0]}_{filter[1]}.dat"
			fle = open(file_out, 'a')
			fle.write("Event_pair correlation_cc lag_time st1 st2 tdelta_s amplitude1 amplitude2\n")

			for k, unique in enumerate(unique_combinations):
				event1, p1 = prep_data(unique[0])
				event2, p2 = prep_data(unique[1])

				'''Computing delta time in hours between the 2 events'''
				tdiff = abs(event2.stats.starttime - event1.stats.starttime) / 3600
				xtdiff.append(tdiff)

				''' Initializing a figure'''
				'''
				For the alignment of the waveforms, 
				we are considering 0.3 seconds before P and 5 seconds after P.
				This time window includes P, S and coda waves. 
				The bandpass filter should be magnitude dependent, e.g. Uchida 2019. 
				'''
				try: 
					lag_time, coeff = xcorr_pick_correction(p1, event1, p2, event2, t_before=0.3,
																t_after=5.0, cc_maxlag=1.0, filter="bandpass", 
																filter_options={'freqmin': filter[0], 'freqmax': filter[1]},
																plot=False,)
				except:
					print(f"------ Event pairs: {event1}-{event2} couldn't get cross-correlated ----")
				
				event1_corrected = event1.trim(p1 - (t_before_p), p1 + (t_after_p))
				event2_corrected = event2.trim(p2 - (t_before_p - lag_time), p2 + (t_after_p + lag_time))
				
				'''Filtering the waveforms before passing them to the rolling window function'''

				event1_prep = event1_corrected.filter("bandpass", freqmin=filter[0], freqmax=filter[1])
				event2_prep = event2_corrected.filter("bandpass", freqmin=filter[0], freqmax=filter[1])
				Time = np.arange(0, len(event1_prep)) * event1_prep.stats.delta
				
				''' Checking the length of both seismograms'''
				l1, l2 = len(event1_prep), len(event2_prep)

				if l1 != l2: 
					_msg = f"Waveforms have different window lenght: {l1}-{l2}, please check the data"
					#raise IOError(_msg)
	 				
					continue

				else:
					cc = np.around(coeff, 3)
					lag = np.around(lag_time, 3)
					print(f"event pair: {k}, correlation: {cc}, lag time: {lag}")
					print(f"event 1: {event1.stats.starttime} <---> event 2: {event2.stats.starttime}")
					delta = event2.stats.starttime - event1.stats.starttime
					name1 = event1.stats.starttime
					name2 = event2.stats.starttime
					amp1 = event1.max()
					amp2 = event2.max()
					fle.write(f"{k} {cc} {lag} {event1.stats.starttime} {event2.stats.starttime} {delta} {amp1} {amp2}\n")
					Relative_time.append(delta)
					print(f"Waveforms have the same length: {l1}. About to enter de rolling window function")
					
				
				
				corr_vals, eq_dist = rolling_window(
								a=event1_prep, 
								b=event2_prep, 
								wlen=windowing[0], 
								stp=windowing[1])

				mean_dist = np.mean(eq_dist)
				std_dist = np.std(eq_dist)
				Relative_distance.append(mean_dist)
				Relative_distance_std.append(std_dist)

				print(f"The mean distance between the two earthquakes is {mean_dist:.6f} m")
				print("-------------------------------------------------------------------------")
				
				corr_vals = np.array(corr_vals)
				xcorr_max.append(corr_vals)

				'''x-Vector for corr-vals'''
				xvals = np.linspace(0, np.max(Time), len(corr_vals))
				xvals_all.append(xvals)
				
			'''Calculating and standard deviation'''
			xcorr_max = np.array(xcorr_max)
			mean_corr = xcorr_max.mean(axis=0)
			stds_corr = xcorr_max.std(axis=0)

			Xcorr.append(mean_corr)
			Xvals.append(xvals_all[0])
			Xstds.append(stds_corr)

			
	processing(station_name=station_name,
				data_list=data_list, 
				t_before_p=trace_len[0], 
				t_after_p=trace_len[1],
				bandpass_filter=bandpass_filter,
				windowing=windowing
				)






