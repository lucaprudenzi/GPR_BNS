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



html_template = """
<body>
    <h1>Hello, {first_header:}!</h1>
    <p>{p2:}, {p1:}!</p>
</body>
"""
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

<h1>Catalogue hybrid waveforms </h1>
In this page
 <ul>
  <li>time domain hybrid waveform with 2 cycles of alignment, numerical relativity data and v4Tsurrogate </li>
  <li>Mismatch for different initial frequencies between v4Tsurrogate and v4Tsurrogate with lambda1,2=0, and between v4Tsurrogate and hybrid waveforms</li>
  <li>Frequency domain amplitude for hybrid, v4Tsurrogate and v4Tsurrogate with lambda1,2=0</li>
  <li>Logarithm of the absolute difference between v4Tsurrogate and hybrid, and between v4Tsurrogate and v4Tsurrogate with lambda1,2=0 </li>
  <li>Frequency domain phase for hybrid, v4Tsurrogate and v4Tsurrogate with lambda1,2=0</li>
  <li>Absolute difference between v4Tsurrogate and hybrid, and between v4Tsurrogate and v4Tsurrogate with lambda1,2=0 </li>
</ul> 
<a href="different_hybridization_windows.html">Go to different hybrization windows</a>
<div>
<table border=1 width=80%>
<tr>
	<th>BAM number</th>
	<th colspan="2">PLOT</th>
</tr>
"""

html_loop = """
<tr>
        <td rowspan="3">BAM:{}</td>
	<td colspan="2"><img class="autoResizeImage" src="{}"\></td>
</tr>
</tr>
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
path_h = "TD/v4Tsurr_Hybrid/h_2cycles/"
path_mismatch = "FD/v4Tsurr_v4Tsurrnotidal/mismatch/"
path_amp = "FD/v4Tsurr_v4Tsurrnotidal/amp/"
path_deltaamp = "FD/v4Tsurr_v4Tsurrnotidal/deltaamp/"
path_phase = "FD/v4Tsurr_v4Tsurrnotidal/phase/"
path_deltaphase = "FD/v4Tsurr_v4Tsurrnotidal/deltaphase/"

elements = BAM_number_list()

html_images_block = ""
for BAM in elements:
    BAM_element = BAM_number_conversion(BAM)
    html_images_block += html_loop.format(BAM_element, path_h+BAM_element+".png", path_h+BAM_element+".png", path_mismatch+BAM_element+".png", path_amp+BAM_element+".png", path_deltaamp+BAM_element+".png", path_phase+BAM_element+".png", path_deltaphase+BAM_element+".png")

index = html_head+html_images_block+html_end
f = open('index.html', 'w')
f.write(index)
f.close()
