# Needed for TC_MALLOC fix
sudo apt-get -f install -y
sudo apt-get install -y google-perftools

# Install Runtime dependencies
sudo dpkg -i /shared/czhenguo/Projects/fruitstand/pipeline_artifacts/aws-neuronx-runtime-lib-2.x.21053.0-47c775bff.deb
sudo dpkg -i /shared/czhenguo/Projects/fruitstand/pipeline_artifacts/aws-neuronx-collectives-2.x.22530.0-1814a2670.deb
if ! apt list 2>/dev/null | grep -q "^aws-neuronx-dkms/now 2.x.4125.0 amd64 \[installed,local\]"; then sudo dpkg -i --force-all /shared/czhenguo/Projects/fruitstand/pipeline_artifacts/aws-neuronx-dkms_2.x.4290.0_amd64.deb; fi
CHECK_STATUS=$?
if [ $CHECK_STATUS -ne 0 ]; then
    echo "Driver version check failed! Terminating job."
    exit 1
fi

# TC MALLOC HACK
LIBTCMALLOC=$(find /usr/lib/x86_64-linux-gnu -name "libtcmalloc.so.*" | sort -V | tail -n 1)
 
if [ -n "$LIBTCMALLOC" ]; then
    # Create a symbolic link to the found libtcmalloc version
    sudo ln -sf "$LIBTCMALLOC" /usr/lib/libtcmalloc.so
    echo "Symbolic link created: /usr/lib/libtcmalloc.so -> $LIBTCMALLOC"
 
    # Export LD_PRELOAD
    export LD_PRELOAD=/usr/lib/libtcmalloc.so
    echo "LD_PRELOAD set to: $LD_PRELOAD"
else
    echo "Error: libtcmalloc.so not found"
    exit 1
fi
