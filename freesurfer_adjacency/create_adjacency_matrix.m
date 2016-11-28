fstarg={'fsaverage5','fsaverage'};
hemi={'lh','rh'};
dest='/data1/vbeliveau/atlas/analyses/clustering/data';
assert_dir_create(dest);

for ntarg=1:numel(fstarg)
    for nh=1:numel(hemi)
        clear L sL A sA f v;
        [~,f]=freesurfer_read_surf([getenv('FREESURFER_HOME') '/subjects/' fstarg{ntarg} '/surf/' hemi{nh} '.pial']);
        
        v=unique(f(:));
        N=numel(v);
        
        % Note: A and L are created separately due to their shear size (especially for fsaverage)
        
        L=zeros(N,N);
        
        for n=v'
            if mod(n,1000)==0
                disp(['Processing vertice ' num2str(n) '/' num2str(N)]);
                pause(0.01);
            end
            indx=f((f(:,1)==n),[2 3]);
            indy=f((f(:,2)==n),[1 3]);
            indz=f((f(:,3)==n),[1 2]);
            
            m=unique([indx(:); indy(:); indz(:)]);
            L(n,m)=-1;
            L(n,n)=numel(m);
        end
        
        sL=sparse(L); % Very large, so save only sparse version
        
        save([dest '/laplacian.' fstarg{ntarg} '.' hemi{nh} '.mat'],'L','-v7.3');
        save([dest '/laplacian.sparse.' fstarg{ntarg} '.' hemi{nh} '.mat'],'sL');        
        clear L sL
        
        A=zeros(N,N);
        for n=v'
            if mod(n,1000)==0
                disp(['Processing vertice ' num2str(n) '/' num2str(N)]);
                pause(0.01);
            end
            indx=f((f(:,1)==n),[2 3]);
            indy=f((f(:,2)==n),[1 3]);
            indz=f((f(:,3)==n),[1 2]);
            
            m=unique([indx(:); indy(:); indz(:)]);
            A(n,m)=1;
        end
                
        sA=sparse(A);
        save([dest '/adjacency.' fstarg{ntarg} '.' hemi{nh} '.mat'],'A');
        save([dest '/adjacency.sparse.' fstarg{ntarg} '.' hemi{nh} '.mat'],'sA');
    end
end