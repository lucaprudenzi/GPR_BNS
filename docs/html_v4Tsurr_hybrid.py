import glob 
import numpy as np

def BAM_number_list():
    elements = np.arange(1, 128, 1)
    #study 56 
    elements = elements[(elements<56)|(elements>57)] 
    elements = elements[(elements<74)|(elements>87)] # very high unf
    elements = elements[(elements<91)|(elements>91)] # precessing
    elements = elements[(elements<93)|(elements>94)] # precessing
    elements = elements[(elements<110)|(elements>119)] # very high unf
    return elements

def BAM_number_conversion(i):
    if i==0:
        i=i+1
    if i < 10:
        BAM_number = "000"+str(i)
    elif i<100:
        BAM_number = "00"+str(i)
    else:
        BAM_number = "0"+str(i) 
    return BAM_number



html_head= """
<html>
<head>
<style>
	.autoResizeImage {
		max-width: 100%;
		max-height: 100%;
	}
</style>
</head>
<body>
<title></title>
In this page there are 3 different choices of hybridization window lenght (0.5 cycles, 2 cycles, 3 cycles) and same starting position of the window (500M from initial point of NR data) for almost all the cases (the shorter waveforms start at 300M)
 <ul>
  <li>3 time domain hybrid waveforms, numerical relativity data and v4Tsurrogate waveform</li>
  <li>Mismatch between v4Tsurrogate and the 3 hybrid waveforms with different initial frequencies </li>
  <li>Mismatch between the 3 hybrid waveforms different initial frequencies </li>
  <li>Frequency domain amplitude for the 3 hybrid waveforms and for v4Tsurrogate </li>
  <li>Logarithm of the absolute difference between v4Tsurrogate and the 3 hybrids </li>
  <li>Frequency domain phase for hybrid, v4Tsurrogate</li>
  <li>Absolute difference between v4Tsurrogate and the 3 hybrids </li>
</ul> 

<div>
<table border=1 width=80%>
<tr>
	<th>BAM number</th>
	<th colspan="2">PLOT</th>
</tr>
"""

html_loop = """
<tr>
        <td rowspan="4">BAM:{}</td>
	<td colspan="2" align="center"><img class="autoResizeImage" src="{}"\></td>
</tr>
<tr>
	<td><img class="autoResizeImage" src="{}"\></td>
	<td><img class="autoResizeImage" src="{}"\></td>
</tr>
<tr>
	<td><img class="autoResizeImage" src="{}"\></td>
	<td><img class="autoResizeImage" src="{}"\></td>
</tr>
<tr>
	<td><img class="autoResizeImage" src="{}"\></td>
	<td><img class="autoResizeImage" src="{}"\></td>
</tr>
"""

html_end = """
</table>

</div>
</body>
</html>
"""
path_h = "TD/v4Tsurr_Hybrid/h/"
path_mismatch = "FD/v4Tsurr_Hybrid/mismatch/"
path_mismatch_hybrid = "FD/v4Tsurr_Hybrid/mismatch_hyb/"
path_amp = "FD/v4Tsurr_Hybrid/amp/"
path_deltaamp = "FD/v4Tsurr_Hybrid/deltaamp/"
path_phase = "FD/v4Tsurr_Hybrid/phase/"
path_deltaphase = "FD/v4Tsurr_Hybrid/deltaphase/"

elements = BAM_number_list()
html_images_block = ""
for BAM in elements:
    BAM_element = BAM_number_conversion(BAM)
    print(BAM_element)
    html_images_block += html_loop.format(BAM_element, path_h+BAM_element+".png", path_mismatch+BAM_element+".png", path_mismatch_hybrid+BAM_element+".png", path_amp+BAM_element+".png", path_deltaamp+BAM_element+".png", path_phase+BAM_element+".png", path_deltaphase+BAM_element+".png")


index = html_head+html_images_block+html_end
f = open('different_hybridization_windows.html', 'w')
f.write(index)
f.close()
