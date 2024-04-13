import pymysql

# 数据库连接信息
host = '38.147.173.234'
user = 'root'
password = '123456'
db = 'training_statistics_db'

try:
    # 尝试连接到数据库
    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=password,
                                 database=db,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    print("数据库连接成功！")
    
    # 添加一个新用户到users表
    with connection.cursor() as cursor:
        # 准备SQL语句
        sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
        # 执行SQL语句
        cursor.execute(sql, ('caiye', '123456'))
        # 提交到数据库执行
        connection.commit()
        print("成功添加用户caiye到数据库。")
    
    # 获取数据库版本信息
    with connection.cursor() as cursor:
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print("数据库版本：", version)
    
finally:
    # 关闭数据库连接
    connection.close()