for d in **/*; do
	if [ ! -d "$d" ]; then
  		mv "$d" "$d.txt"
fi
done

mkdir ~/Desktop/master
mv */*.txt ~/Desktop/master/
