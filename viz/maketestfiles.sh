outfile=./testfiles.js
infiles=../tests/fixtures/*.json

echo "creating new $outfile"
echo "" > $outfile
echo "var testFiles = [];" >> $outfile

for file in $infiles; do
	fname=`basename $file`
	echo "adding $fname to $outfile"
	echo "testFiles[\"$fname\"] = " >> $outfile
	cat $file >> $outfile
	echo ";" >> $outfile
done

echo "new $outfile created from files in $infiles"
