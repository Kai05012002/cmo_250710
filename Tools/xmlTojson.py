import json
import xmltodict
import re

def convert_scenario_to_json():
    # 读取原始文件
    print("test")
    with open('C:/Program Files (x86)/Steam/steamapps/common/Command - Modern Operations/ImportExport/T3_C3.inst', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 解析外层JSON
    outer_json = json.loads(content)
    print("test")
    # 提取Comments中的XML内容
    xml_content = outer_json['Comments']
    
    # 将XML转换为字典
    xml_dict = xmltodict.parse(xml_content)
    
    # 将转换后的字典转为JSON字符串
    scenario_json = json.dumps(xml_dict, indent=2, ensure_ascii=False)
    print("test")
    # 保存转换后的JSON到新文件
    with open('T3_C3.json', 'w', encoding='utf-8') as outfile:
        outfile.write(scenario_json)
    
    return xml_dict

if __name__ == "__main__":
    try:
        result = convert_scenario_to_json()
        print("转换成功！结果已保存到 scenario.json")
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")