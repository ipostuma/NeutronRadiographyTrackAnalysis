MYDIR=$1
FNAME=$2

if [ ! -d $MYDIR"converted/" ]; then
  ./convert.sh $MYDIR
fi
