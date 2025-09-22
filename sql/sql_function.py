import mysql.connector

con = mysql.connector.connect(
    user="root",
    password="6256875",
    host="127.0.0.1",
    database="mydb"
)
print("con")
cursor = con.cursor()
#cursor.execute("INSERT INTO mysql.users(uid, email) VALUES (%s, %s)", (user.userid, user.email))
cursor.execute("INSERT INTO users(uid, email) VALUES ('qwe', 'test@gmail.com')")
con.commit()
con.close()