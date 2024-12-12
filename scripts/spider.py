import requests
from bs4 import BeautifulSoup
import urllib.parse
import xmltodict
import json

# 主程序
if __name__ == '__main__':
    # 输入 URL 和参数
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    import time
    
    options = Options()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')  # 使用隐身模式（可选）
    options.add_argument('--headless')  # 启用无头模式
    # options.add_argument('--disable-gpu')  # 禁用 GPU 加速
    # options.add_argument('--disable-software-rasterizer')  # 禁用软件光栅化
    # options.add_argument('--disable-webgl')  # 禁用 WebGL
    # options.add_argument('--no-sandbox')  # 无沙盒模式（可选）
    # options.add_argument('--disable-features=MediaRouter')
    
    # 设置 WebDriver
    service = Service(
        executable_path='D:/Program2/ChromeDriver/chromedriver-win64/chromedriver.exe'
    ) 
    # 启动 Chrome 浏览器
    driver = webdriver.Chrome(service=service, options=options)
    
    query = "x+2y+z=6, 2x-y+3z=14, y+2z=8"
    query = urllib.parse.quote(f"{query}")
    url = f"https://www.symbolab.com/solver/step-by-step/{query}?or=input"
    print(url)
    
    driver.get(url)
    time.sleep(3)
    
    html_content = driver.page_source

    soup = BeautifulSoup(html_content, 'html.parser')
    content_divs = soup.find_all(
        'div', 
        class_='solution_step_list'
    )[0]
    driver.quit()

    step_text_list = {}
    for i, child in enumerate(content_divs.children):
        if child.name:
            if child['class'][0] == 'solution_step_title':
                step_i_exprs = []
                expr_tr = child.find_all('tr')
                for tr in expr_tr:
                    step_i_exprs.append(tr.text)
                if len(step_i_exprs) == 0:
                    step_i_exprs = [child.text]
                    
                step_text_list[i] = step_i_exprs
                print(f"Element: {child.name}, Content: {step_i_exprs}")
            else:
                step_text_list[i] = [child.text]
                    
                print(f"Element: {child.name}, Content: {child.text.strip()}")
            
    print(json.dumps(step_text_list, indent=4, ensure_ascii=False))
            
        
        
    with open('content_divs.html', 'w', encoding='utf-8') as f:
        for div in content_divs:
            f.write(str(div) + '\n')  # 将 div 的 HTML 内容写入文件
            
    with open('symbolab.html', 'w', encoding='utf-8') as file:
        file.write(soup.prettify())

    
