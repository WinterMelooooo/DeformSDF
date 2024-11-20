unset LD_LIBRARY_PATH
# 定义配置文件和脚本路径
CONF_FILE="./confs/dtu_mine_66.conf"
DATA_DIR="../data/DTU_from_4D_20240709_ypzhao_resized"
PYTHON_SCRIPT="./training/exp_runner.py"
START_ID=1
END_ID=99

for scan_id in $(seq $START_ID $END_ID)
do
    # 创建一个临时配置文件
    TEMP_CONF_FILE="./confs/dtu_mine_66_scan_${scan_id}.conf"
    cp $CONF_FILE $TEMP_CONF_FILE

    # 修改临时配置文件中的 scan_id
    sed -i "s/^\(\s*scan_id\s*=\s*\).*$/\1${scan_id} #TODO/" $TEMP_CONF_FILE

    # 执行 Python 脚本
    python $PYTHON_SCRIPT --conf ${TEMP_CONF_FILE} --scan_id ${scan_id}

    # 删除临时配置文件
    rm $TEMP_CONF_FILE
done
