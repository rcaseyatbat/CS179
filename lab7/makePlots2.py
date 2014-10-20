# Credits to Jeff Amelang, 2013

import os

continueConsolidating = True

fileIndex = 0
while (continueConsolidating == True):
  baseFilename = 'output/displacements_%08d' % fileIndex
  consolidatedFilename = baseFilename + '.dat'
  rank0Filename = baseFilename + '_%03d.dat' % 0
  if (os.path.exists(rank0Filename)):
    consolidatedFile = open(consolidatedFilename, 'w')
    processingFiles = True
    rank = 0
    while (processingFiles == True):
      rankFilename = baseFilename + '_%03d.dat' % rank
      if (os.path.exists(rankFilename)):
        #print 'reading file %s\n' % rankFilename
        rankFile = open(rankFilename, 'r')
        for line in rankFile:
          consolidatedFile.write(line)
        rankFile.close()
        os.system('rm -f %s' % rankFilename)
        rank = rank + 1
      else:
        processingFiles = False
    consolidatedFile.close()
  if (os.path.exists(consolidatedFilename)):
    gnuplotFile = open('output/gnuplotScript.gnuplot', 'w')
    gnuplotFile.write('set datafile separator ","\n')
    gnuplotFile.write('plot "%s" using 1:2 with lines lw 2 lc 1 lt 1 title "displacements"\n' % consolidatedFilename)
    gnuplotFile.write('set grid\n')
    gnuplotFile.write('set xrange [0:1]\n')
    gnuplotFile.write('set yrange [-2:2]\n')
    gnuplotFile.write('set terminal png\n')
    gnuplotFile.write('set output "%s.png"\n' % os.path.splitext(consolidatedFilename)[0])
    gnuplotFile.write('set key right box\n')
    gnuplotFile.write('set ylabel "Displacement"\n')
    gnuplotFile.write('set xlabel "Position"\n')
    gnuplotFile.write('replot\n')
    gnuplotFile.close()
    os.system('gnuplot output/gnuplotScript.gnuplot')
    fileIndex = fileIndex + 1
  else:
    continueConsolidating = False
