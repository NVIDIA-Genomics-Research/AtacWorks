Pretrained_models
=================

This is a list of AtacWorks denoising models that were reported in Lal & Chiang, 2019. 
Models were trained to denoise bulk or single-cell ATAC-seq data of varying sequencing depths.

The training data for these models was generated from BAM files using `this script <https://github.com/zchiang/atacworks_analysis/blob/master/preprocessing/atac_bam2bw.sh>`_. 

Briefly, each ATAC-seq read was converted to a single genomic position corresponding to the first base pair of the read. Reads aligning to the + strand were offset by +4 bp, while reads aligning to the - strand were offset by -5 bp. Each cut site location was extended by 100 bp in either direction. The bedtools genomecov function was used to convert the list of locations into a coverage track. To call peaks from clean and noisy signal tracks, MACS2 subcommands bdgcmp and bdgpeakcall were run with the ppois parameter and a -log10(p-value) cutoff of 3. BED files with equal coverage over all chromosomes were provided as a control input track.

Before using these models, please note that they should only be used on data that is processed exactly as described above. If your bigWig files and/or peak calls are generated using a different method, these models may give unreliable results, and it would be preferable to train your own model as described in Tutorial 1. 


+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Data type   |Noise type     |Noisy data depth |Clean data depth |Noisy data cell count |Clean data cell count |Single-cell protocol |Training cell type(s) |Interval size (bp) |Model path |
+============+===============+=================+=================+======================+======================+=====================+======================+===================+===========+
|Bulk        |Low coverage   |0.2M             |50M              |N/A                   |N/A	               |N/A	             |CD4, CD8, B, NK       |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Bulk	     |Low coverage   |1M               |50M              |N/A                   |N/A	               |N/A	             |CD4, CD8, B, NK       |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Bulk        |Low coverage   |5M               |50M              |N/A                   |N/A	               |N/A	             |CD4, CD8, B, NK       |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Bulk        |Low coverage   |10M	       |50M              |N/A                   |N/A	               |N/A	             |CD4, CD8, B, NK       |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Bulk        |Low coverage   |20M              |50M              |N/A                   |N/A	               |N/A	             |CD4, CD8, B, NK       |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Bulk        |Low quality    |20M              |20M              |N/A                   |N/A	               |N/A	             |Monocytes             |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Single-cell |Low cell count |~0.2M            |~13M             |90                    |6000	               |dsci-ATAC            |CD4, CD8, preB        |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Single-cell |Low cell count |~1M              |~13M             |450                   |6000	               |dsci-ATAC            |CD4, CD8, preB        |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Single-cell |Low cell count |~0.2M            |~48M             |10                    |2400	               |dsc-ATAC             |B, Monocytes          |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+
|Single-cell |Low cell count |~1M              |~48M             |50                    |2400	               |dsc-ATAC             |B, Monocytes          |50,000             |           |
+------------+---------------+-----------------+-----------------+----------------------+----------------------+---------------------+----------------------+-------------------+-----------+

