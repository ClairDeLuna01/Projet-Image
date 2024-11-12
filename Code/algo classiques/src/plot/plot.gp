set terminal pngcairo size 800,600
set output 'src/plot/out/gaussianFilterPlot.png'
set title "Évolution du PSNR en dB en fonction du sigma du filtre gaussien"
set xlabel "Evolution du sigma"
set ylabel "PSNR (dB)"
set grid
plot "src/plot/dat/gaussianFilterSigma.dat" using 1:2 with linespoints title "PSNR en dB"
