clc;
clear all;

#### Information Gain Ratio
SVM_1 = [[0.24296414], [0.17879691], [0.15496806], [0.22695777], [0.11544953], [0.13603666], [0.20007369], [0.23313171], [0.13393677], [0.19444644], [0.08275924], [0.17271221]]';
SVM_2 = [[0.24203354], [0.22357034], [0], [0.09979148], [0.06133819], [0.00954166], [0.04194425], [0.1011497], [0], [0.35782052], [0.02678552], [0.27623059]]';

Dagging_1 = [[0.22655728], [0.16813735], [0.1512259], [0.2377874], [0.10942882], [0.12124996], [0.20641475], [0.25082647], [0.13294509], [0.19866023], [0.01930436], [0.17458989]]';
Dagging_2 = [[0.24203354], [0.22357034], [0], [0.10812481], [0.06133819], [0.00954166], [0.04194425], [0.1011497], [0], [0.35782052], [0.01492856], [0.25799584]]';

x = [1:1:12]';

avg_val = [];
lerr = [];
uerr = [];
for i = 1:size(SVM_1,1)
  mat_val = [SVM_1(i) SVM_2(i) Dagging_1(i) Dagging_2(i)]';
  avg_val = [avg_val; mean(mat_val)];
  lerr = [lerr; avg_val(i) - min(mat_val)];
  uerr = [uerr; -1*avg_val(i) + max(mat_val)];
endfor
bar(x, avg_val);
hold on;
errorbar(x, avg_val, lerr, uerr);
#xlabel({'Elevation  Slope  Aspect  Total Curvature TRI TWI SPI STI Rainfall Stream Distance Soil Type LULC'});
ylabel('Information Gain Ratio');
title('Information Gain Ratio');
set(gca, 'XTickLabel',{'Elevation', 'Slope', 'Aspect', 'Total Curvature', ...
      'TRI', 'TWI', 'SPI', 'STI', ...
      'Rainfall', 'Stream Distance', 'Soil Type', 'LULC'});
##xtl = xticklabels ();
##xtl = ['Elevation', 'Slope', 'Aspect', 'Total Curvature', ...
##      'Roughness Index', 'Wetness Index', 'Stream Power Index', 'Sediment Trasport', ...
##      'Rainfall', 'Distance to nearby stream', 'Soil Type', 'LULC']';
##xt = xticks ();
##for ii = 1:numel (xtl)
##  text (xt(ii), 100, xtl{ii}, "rotation", 45, "horizontalalignment", "right")
##endfor
##xticklabels ([]);
set(gca, "fontsize", 16);