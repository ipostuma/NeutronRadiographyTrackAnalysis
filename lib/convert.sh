myDir="converted"
#creo la cartella converted in $1
if [ -d "$1$myDir" ]; then
  rm $1$myDir/*
else
  mkdir $1$myDir
fi
#faccio la conversione dei file in jpg
#convert $1 -sharpen 2 -resize 1200x -quality 40 $2$3
for pic in $1*tif;
do
  echo $pic
  filename=`basename "$pic"`
  filename="${filename%.*}"
  echo $filename
  convert "$pic" -sharpen 2 -resize 1200x -quality 80 "$1$myDir/$filename.jpg"
done
