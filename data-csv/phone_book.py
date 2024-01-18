import csv

# with open('data/phone_book.csv', mode='r') as csv_file:
#     # next(csv_file)
#     csv_reader = csv.reader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         print(row)
#         if line_count > 0:
#             print(f"{row[1]}: {row[2]}")
#         # line_count += 1


with open('data/phone_book.csv', mode='r') as csv_file:
    csv_reader  = csv.DictReader(csv_file)
    for row in csv_reader:
        # print(row)
        print(f"{row['last_name']}: {row['phone_number']}")
