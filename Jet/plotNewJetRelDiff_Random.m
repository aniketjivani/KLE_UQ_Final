clear;
clc;
close all;

%% Load colorblind colormap
load('./colorblind_colormap/colorblind_colormap.mat', 'colorblind', 'colornames');

%%

set(0, 'DefaultAxesLineWidth', 2);
set(0, 'DefaultAxesFontSize', 30, 'DefaultAxesFontName', 'Times');


errTest = load("/Users/ajivani/Downloads/JetRandomErrs.mat");
errTest = cell2mat(errTest.test_errors);

ipTest = load("/Users/ajivani/Downloads/JetTestPts2.mat");
ipTest = ipTest.test_points;

ipHF = load("/Users/ajivani/Downloads/JetRandomPts.mat");
ipHF = ipHF.HFPoints;

% if bt == 2
%     [errValsSorted, errIndices] = sort(err02);
% elseif bt == 3
%     [errValsSorted, errIndices] = sort(err03);
% elseif bt == 4
%     [errValsSorted, errIndices] = sort(err04);
% elseif bt == 5
%     [errValsSorted, errIndices] = sort(err05);
% elseif bt == 6
%     [errValsSorted, errIndices] = sort(err06);
% elseif bt == 7
%     [errValsSorted, errIndices] = sort(err07);
% elseif bt == 8
%     [errValsSorted, errIndices] = sort(err08);
% end
trainPtsHF = ipHF(:, :);

[errValsSorted, errIndices] = sort(errTest);

ipTestSorted = ipTest(errIndices, :);

dpi_scale = 300 / 200;

figure(); 
scatter(ipTestSorted(:, 1), ipTestSorted(:, 2), 100, errValsSorted, 'filled', 'DisplayName', 'Relative difference');
scatterCB = colorbar;

% set caxis in original scale of relative difference, not log scale.
% % caxis([0.0008 0.018])
caxis([0.002 0.65])

% caxis([-7 -4])
% set(get(scatterCB, 'Title'),  'String', 'Relative Difference');
hold on;
% 
% ipListTrain = [aVec bVec];
% 
% % trainIdxHF = 1:size(YHF, 2);
% trainIdxHF = 1:size(YCT, 2);
% 
% 
% % scatter(ipListTrain(trainIdxHF, 1), ipListTrain(trainIdxHF, 2), 55, colorblind(7, :), 'filled', 'd', 'DisplayName', 'Training Points HF');
scatter(trainPtsHF(:, 1), trainPtsHF(:, 2), 200, colorblind(7, :), 'filled', 'd', 'DisplayName', 'Training Points MF');

% legend('Location', 'bestoutside', 'Orientation', 'horizontal');
% hold on;

% ylabel('$\kappa$', 'Interpreter', 'latex');
xlabel('$U_c$ [m/s]', 'Interpreter', 'latex', 'FontSize', 30);


% xlim([292 314]);
% ylim([0.1 0.3]);

xlim([291.95 314.05]);
ylim([0.095, 0.305]);

box on;

set(gcf, 'Color', [1, 1, 1]);
fig = gcf;
fig.Position(3:4) = [800, 600];
set(fig, 'PaperPositionMode', 'auto');
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperSize', [fig.Position(3)/150, fig.Position(4)/150]); 

% exportgraphics(fig, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/JetRandomAll.jpg",Resolution=300)
% exportgraphics(fig, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/JetRandomAll.eps", 'ContentType','vector')

print(fig, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/JetRandomAll.jpg", '-djpeg','-r300')


% print(fig, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/JetRandomAll.jpg", '-djpeg', '-200')

% title('Relative Difference between HF and MF');
