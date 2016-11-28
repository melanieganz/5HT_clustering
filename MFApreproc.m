%% Preprocess data

info=importdata('/data1/vbeliveau/atlas/lists/info_alltracers_base_healthy_hrrt.mat');

dest='/data1/vbeliveau/MFApreproc';
if ~exist(dest,'dir'), mkdir(dest); end

%% Map TACs to fsaverage5

importPET='/data1/vbeliveau/atlas/import/PET';
procPET='/data2/FSproc/PET';

% tracers={'dasb','sb','az','C36'};
tracers={'cumi'};
hemi={'lh','rh'};
fwhm=[0 5];
for nt=1:numel(tracers)
    % Sample data to TACs fsaverage5
    subjlist=info.petID(cellfun(@(x) ~isempty(regexp(x,['^(' tracers{nt} ')*'])),info.petID)); %#ok<RGXP1>
    if ~exist([dest '/' tracers{nt}],'dir'), mkdir([dest '/' tracers{nt}]); end
    
    for ns=1:numel(subjlist)
        for nh=1:numel(hemi)
            mov=[importPET '/' subjlist{ns} '/tac.realigned.nii.gz'];
            reg=[procPET '/' subjlist{ns} '/tac.realigned.wavg.GD.lta'];
            for nf=1:numel(fwhm)
                out=[dest '/' tracers{nt} '/' subjlist{ns} '.' hemi{nh} '.fsaverage5.sm' num2str(fwhm(nf)) '.nii.gz'];
                if fwhm(nf)~=0, smflag=[' --surf-fwhm ' num2str(fwhm(nf))]; else smflag=''; end
                unix(['mri_vol2surf --mov ' mov ' --reg ' reg ' --hemi ' hemi{nh} ' --o ' out ...
                    ' --trgsubject fsaverage5 --projfrac 0.5 --cortex ' smflag]);
            end
        end
    end
end

%% Concatenate, demean and standardize rows

tracers={'cumi','dasb','sb','az','C36'};
% tracers={'cumi'};

lh_mask=[]; rh_mask=[];
fwhm=[0 5];

for nt=1:numel(tracers)
    data={}; eigvals={};
    
    % Sample data to TACs fsaverage5
    subjlist=info.petID(cellfun(@(x) ~isempty(regexp(x,['^(' tracers{nt} ')*'])),info.petID)); %#ok<RGXP1>
    if ~exist([dest '/' tracers{nt}],'dir'), mkdir([dest '/' tracers{nt}]); end
    
    for nf=1:numel(fwhm)
        for ns=1:numel(subjlist)
            lh=MRIread([dest '/' tracers{nt} '/' subjlist{ns} '.lh.fsaverage5.sm' num2str(fwhm(nf)) '.nii.gz']);
            rh=MRIread([dest '/' tracers{nt} '/' subjlist{ns} '.rh.fsaverage5.sm' num2str(fwhm(nf)) '.nii.gz']);
            
            % Mask data and remove empty frames
            if isempty(lh_mask), lh_mask=sum(lh.vol,4)~=0; end
            if isempty(rh_mask), rh_mask=sum(rh.vol,4)~=0; end
            valid_frames=sum(lh.vol,2)~=0;
            lh.vol=lh.vol(1,lh_mask,1,valid_frames);
            rh.vol=rh.vol(1,rh_mask,1,valid_frames);
            
            % Demean and store subject data
            data{ns}=detrend([squeeze(lh.vol); squeeze(rh.vol)],'constant');  %#ok<SAGROW>
            data{ns}=data{ns}./repmat(std(data{ns},1,2),1,size(data{ns},2));
            
            % Identify and store eigenvalues
            [U,S,V]=svd(data{ns});            
            
            eigvals{ns}=diag(S); %#ok<SAGROW>
        end
        
        save([dest '/' tracers{nt} '/MFA.preproc.sm' num2str(fwhm(nf)) '.mat'],'data','eigvals','subjlist','lh_mask','rh_mask');
    end
end