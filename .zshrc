
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/auchen/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/auchen/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/auchen/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/auchen/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


# gclould path
export PATH=$PATH:/Users/auchen/google-cloud-sdk/bin
export PATH=$PATH:/Users/auchen/google-cloud-sdk
# The next line updates PATH for the Google Cloud SDK.
if [ -f '/Users/auchen/google-cloud-sdk/path.zsh.inc' ]; then . '/Users/auchen/google-cloud-sdk/path.zsh.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/Users/auchen/google-cloud-sdk/completion.zsh.inc' ]; then . '/Users/auchen/google-cloud-sdk/completion.zsh.inc'; fi

source /Users/auchen/.docker/init-zsh.sh || true # Added by Docker Desktop
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"


