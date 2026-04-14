wget -P ./data https://ndownloader.figshare.com/files/38256840
wget -P ./data https://ndownloader.figshare.com/files/38256858
cd data
unzip 38256840
unzip 38256858
rm -f 38256840
rm -f 38256858
for id in {001..100}; do
    mv ./BSD100/image_SRF_4/img_${id}_SRF_4_HR.png ./BSD100/
done
for id in {001..100}; do
    mv ./Urban100/image_SRF_4/img_${id}_SRF_4_HR.png ./Urban100/
done
rm -rf ./BSD100/image_SRF_2
rm -rf ./BSD100/image_SRF_3
rm -rf ./BSD100/image_SRF_4
rm -rf ./Urban100/image_SRF_2
rm -rf ./Urban100/image_SRF_4
rm -f ./Urban100/source_selected.xlsx
cd ..
