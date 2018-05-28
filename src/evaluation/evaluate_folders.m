function evaluate_folders(reference_dir,degraded_dir,output_path)

reference_dir = [reference_dir '/'];
degraded_dir = [degraded_dir '/'];

reference_filenames = dir([reference_dir '*.wav']);
%degraded_filenames = dir([degraded_dir '*.wav']);
[filepath,~,~] = fileparts(output_path);
[~, ~, ~] = mkdir(filepath);
filenames = {reference_filenames.name};

data_STOI = zeros(length(reference_filenames),1);
data_eSTOI = zeros(length(reference_filenames),1);
data_SDR = zeros(length(reference_filenames),1);

for i=1:length(filenames)
    filename = filenames{i};
    rfp = [reference_dir filename];
    dfp = [degraded_dir filename];

    [x, fs_r] = audioread(rfp);
    [xh, fs_d] = audioread(dfp);
    assert(fs_r == fs_d)
    fs = fs_r;

    x = transpose(x);
    xh = transpose(xh);

    try
    data_STOI(i) = stoi(x,xh,fs);
    catch
    fprintf(['Problem using stoi.  Assigning a value of NaN ' filename]);
    data_STOI(i) =  NaN;
    end

    try
    data_eSTOI(i) = estoi(x,xh,fs);
    catch
    fprintf(['Problem using estoi.  Assigning a value of NaN ' filename]);
    data_eSTOI(i) = NaN;
    end

    try
    [SDR,~,~,~] = bss_eval_sources(xh,x);
    data_SDR(i) = SDR;
    catch
    fprintf(['Problem using SDR.  Assigning a value of NaN ' filename]);
    data_SDR(i) = NaN;
    end
end

save(output_path, 'data_STOI','data_eSTOI','data_SDR','filenames','reference_dir','degraded_dir')
fprintf(output_path)
end
