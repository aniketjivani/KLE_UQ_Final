clear;
clc;
close all;

%%
bt = 6;

%% Load colorblind colormap
load('./colorblind_colormap/colorblind_colormap.mat', 'colorblind', 'colornames');

%%

set(0, 'DefaultAxesLineWidth', 2);
set(0, 'DefaultAxesFontSize', 30, 'DefaultAxesFontName', 'Times');


errTest = load("/Users/ajivani/Downloads/BatchErrors.mat");

err02 = cell2mat(errTest.testErr2);
err03 = cell2mat(errTest.testErr3);
err04 = cell2mat(errTest.testErr4);
err05 = cell2mat(errTest.testErr5);
err06 = cell2mat(errTest.testErr6);
err07 = cell2mat(errTest.testErr7);
err08 = cell2mat(errTest.testErr8);

ipTest = load("/Users/ajivani/Downloads/JetTestPts2.mat");
ipTest = ipTest.test_points;

ipHF = load("/Users/ajivani/Downloads/JetHFPts2.mat");
ipHF = ipHF.HFPoints;


if bt == 2
    [errValsSorted, errIndices] = sort(err02);
elseif bt == 3
    [errValsSorted, errIndices] = sort(err03);
elseif bt == 4
    [errValsSorted, errIndices] = sort(err04);
elseif bt == 5
    [errValsSorted, errIndices] = sort(err05);
elseif bt == 6
    [errValsSorted, errIndices] = sort(err06);
elseif bt == 7
    [errValsSorted, errIndices] = sort(err07);
elseif bt == 8
    [errValsSorted, errIndices] = sort(err08);
end
trainPtsHF = ipHF(1:(15 + (bt - 1) * 5), :);
    
ipTestSorted = ipTest(errIndices, :);

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


% scatter(trainPtsHF(:, 1), trainPtsHF(:, 2), 80, colorblind(7, :), 'filled', 'd', 'DisplayName', 'Training Points MF');
scatter(trainPtsHF(:, 1), trainPtsHF(:, 2), 200, colorblind(7, :), 'filled', 'd', 'DisplayName', 'Training Points MF');

% legend('Location', 'bestoutside', 'Orientation', 'horizontal');
% hold on;

ylabel('$\kappa$', 'Interpreter', 'latex', 'FontSize', 30);
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
% exportgraphics(fig, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/JetALAllBatch" + num2str(bt - 1) + ".jpg",Resolution=300)

print(fig, "/Users/ajivani/Desktop/Research/KLE_UQ_Final/Jet/figs_jet/JetALAllBatch" + num2str(bt - 1) + ".jpg", '-djpeg','-r300')

% title('Relative Difference between HF and MF');
